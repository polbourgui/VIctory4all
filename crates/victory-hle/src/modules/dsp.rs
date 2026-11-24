//! The console's audio DSP (`chad.dll`): streams that convolve blocks of
//! float samples with an impulse response (reverb), or run forward and
//! inverse FFTs over them, on a coprocessor.
//!
//! A title connects once with a memory block for the DSP, activates
//! streams, and submits one command per block (a gain ramp, an input and an
//! output buffer). Activation hands back a status block whose second word is
//! the ticket of the last completed command and whose fourth is its result;
//! the title checks it one block later and treats a late command as an
//! overrun. Commands complete when submitted here.
//!
//! We compute each command when it is submitted, with
//! [`victory_formats::xdsp`]: convolution takes the impulse response in
//! partitions of one block's worth of complex values, and the transforms
//! read and write the same packed spectra.
// shortcut: the hardware's spectrum layout is inferred (see `xdsp`). If a
// title's reverb sounds wrong while it ships its responses already
// transformed, the packing, bin order or scaling there is the suspect.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;

use victory_formats::xdsp::{self, Convolver, Transform};

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;

const CONVOLUTION: u32 = 1;
const FORWARD_FFT: u32 = 2;
const INVERSE_FFT: u32 = 3;
/// Stereo input and output come as one block per channel.
const DEINTERLEAVED: u32 = 2;
/// Spectra in magnitude and phase; transforms only.
const POLAR: u32 = 4;
/// Two impulse responses, left then right.
const STEREO_RESPONSE: u32 = 8;
/// Option bits that need a stereo stream.
const STEREO_OPTIONS: u32 = DEINTERLEAVED | STEREO_RESPONSE;

pub struct Dsp {
    max_streams: u32,
    active: Mutex<u32>,
}

/// What activation describes.
#[repr(C, packed)]
pub struct StreamDesc {
    kind: u32,
    options: u32,
    frames: u32,
    channels: u32,
    ir_floats: u32,
    ir: *const f32,
}

/// The status block a stream's owner polls.
#[repr(C)]
pub struct Status {
    submitted: u32,
    completed: u32,
    reserved: u32,
    result: HRESULT,
}

enum Work {
    Convolution {
        convolver: Convolver,
        deinterleaved: bool,
    },
    /// No impulse response to convolve with: silence.
    Silence,
    Forward {
        transform: Transform,
        polar: bool,
    },
    Inverse {
        transform: Transform,
        polar: bool,
    },
}

pub struct Stream {
    dsp: *mut Dsp,
    frames: u32,
    channels: u32,
    work: Work,
    status: Box<Status>,
}

/// One block of work.
#[repr(C)]
pub struct Command {
    /// The output's gain at the first frame, ramping towards `end_scale`
    /// (convolution only).
    begin_scale: f32,
    end_scale: f32,
    input: *const f32,
    output: *mut f32,
    /// Multiplies the input before a forward transform.
    multiplier: *const f32,
}

pub extern "system" fn XDspConnectWithMaximumStreamLimit(
    base: *mut c_void,
    base_length: u32,
    ir_seconds: u32,
    max_streams: u32,
    max_channels: u32,
    out: *mut *mut Dsp,
) -> HRESULT {
    if base.is_null()
        || !(0x1_0000..=0x1F40_0000).contains(&base_length)
        || ir_seconds > 0x80
        || max_streams > 0x100
        || max_channels > 2
    {
        return E_INVALIDARG;
    }
    if out.is_null() {
        return E_POINTER;
    }
    tracing::info!(
        base_length,
        ir_seconds,
        max_streams,
        max_channels,
        "audio DSP connected"
    );
    let dsp = Box::new(Dsp {
        max_streams,
        active: Mutex::new(0),
    });
    // SAFETY: checked non-null.
    unsafe { out.write(Box::into_raw(dsp)) };
    S_OK
}

pub extern "system" fn XDspDisconnect(dsp: *mut Dsp) -> HRESULT {
    if dsp.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspConnectWithMaximumStreamLimit`; disconnected
    // once, after its streams.
    drop(unsafe { Box::from_raw(dsp) });
    S_OK
}

fn valid(desc: &StreamDesc) -> bool {
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    let shape = match kind {
        CONVOLUTION => {
            (1..=2).contains(&channels)
                && matches!(frames, 0x200 | 0x400)
                && options & POLAR == 0
                && !desc.ir.is_null()
        }
        FORWARD_FFT | INVERSE_FFT => channels == 1 && matches!(frames, 0x200 | 0x400 | 0x800),
        _ => false,
    };
    shape && (options & STEREO_OPTIONS == 0 || channels == 2)
}

pub extern "system" fn XDspActivate(
    dsp: *mut Dsp,
    desc: *const StreamDesc,
    status: *mut *const Status,
    stream: *mut *mut Stream,
) -> HRESULT {
    if status.is_null() || stream.is_null() {
        return E_POINTER;
    }
    if dsp.is_null() || desc.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes its description.
    let desc = unsafe { &*desc };
    if !valid(desc) {
        return E_INVALIDARG;
    }
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    tracing::debug!(
        kind,
        options,
        frames,
        channels,
        "audio DSP stream activated"
    );
    {
        // SAFETY: a connection from `XDspConnectWithMaximumStreamLimit`.
        let dsp = unsafe { &*dsp };
        let mut active = dsp.active.lock().unwrap_or_else(|e| e.into_inner());
        if *active >= dsp.max_streams {
            return E_OUTOFMEMORY;
        }
        *active += 1;
    }
    let work = match kind {
        CONVOLUTION => convolution(desc),
        FORWARD_FFT => Work::Forward {
            transform: Transform::new(frames as usize),
            polar: options & POLAR != 0,
        },
        _ => Work::Inverse {
            transform: Transform::new(frames as usize),
            polar: options & POLAR != 0,
        },
    };
    let new = Box::new(Stream {
        dsp,
        frames,
        channels,
        work,
        status: Box::new(Status {
            submitted: 0,
            completed: 0,
            reserved: 0,
            result: S_OK,
        }),
    });
    // SAFETY: checked non-null.
    unsafe {
        status.write(&*new.status);
        stream.write(Box::into_raw(new));
    }
    S_OK
}

/// Copies the title's impulse response into a convolver. Partitions are
/// whole blocks; a remainder is dropped, as the hardware counts them.
fn convolution(desc: &StreamDesc) -> Work {
    let (frames, channels) = (desc.frames as usize, desc.channels as usize);
    let floats = desc.ir_floats as usize;
    let usable = floats / (2 * frames) * (2 * frames);
    let responses = if channels == 2 && desc.options & STEREO_RESPONSE != 0 {
        2
    } else {
        1
    };
    if usable == 0 {
        tracing::warn!(
            floats,
            frames,
            "audio DSP: impulse response shorter than a partition"
        );
        return Work::Silence;
    }
    let mut response = Vec::with_capacity(usable * responses);
    for r in 0..responses {
        // SAFETY: the title's response holds `floats` floats per channel.
        let channel = unsafe { std::slice::from_raw_parts(desc.ir.add(r * floats), usable) };
        response.extend_from_slice(channel);
    }
    match Convolver::new(frames, channels, &response, usable, responses) {
        Some(convolver) => Work::Convolution {
            convolver,
            deinterleaved: desc.options & DEINTERLEAVED != 0,
        },
        None => Work::Silence,
    }
}

pub extern "system" fn XDspDeactivate(stream: *mut Stream) -> HRESULT {
    if stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspActivate`; deactivated once.
    let stream = unsafe { Box::from_raw(stream) };
    // SAFETY: the connection outlives its streams.
    let dsp = unsafe { &*stream.dsp };
    *dsp.active.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
    S_OK
}

/// Runs one block through a stream's work into `output`.
fn run(stream: &mut Stream, command: &Command, input: &[f32], output: &mut [f32]) {
    let frames = stream.frames as usize;
    match &mut stream.work {
        Work::Convolution {
            convolver,
            deinterleaved,
        } => {
            let channels = convolver.channels();
            let (begin, end) = (command.begin_scale, command.end_scale);
            if channels == 1 || *deinterleaved {
                for (c, (i, o)) in input
                    .chunks(frames)
                    .zip(output.chunks_mut(frames))
                    .enumerate()
                {
                    convolver.process(c, i, o);
                    xdsp::ramp(o, frames, 1, begin, end);
                }
            } else {
                let mut lane = vec![0.0; frames];
                let mut out = vec![0.0; frames];
                for c in 0..channels {
                    for (l, frame) in lane.iter_mut().zip(input.chunks(channels)) {
                        *l = frame[c];
                    }
                    convolver.process(c, &lane, &mut out);
                    for (frame, s) in output.chunks_mut(channels).zip(&out) {
                        frame[c] = *s;
                    }
                }
                xdsp::ramp(output, frames, channels, begin, end);
            }
        }
        Work::Silence => output.fill(0.0),
        Work::Forward { transform, polar } => {
            if command.multiplier.is_null() {
                transform.forward(input, output);
            } else {
                // SAFETY: the title's multiplier holds one block.
                let multiplier = unsafe { std::slice::from_raw_parts(command.multiplier, frames) };
                let product: Vec<f32> = input.iter().zip(multiplier).map(|(a, b)| a * b).collect();
                transform.forward(&product, output);
            }
            if *polar {
                xdsp::to_polar(output);
            }
        }
        Work::Inverse { transform, polar } => {
            if *polar {
                let mut cartesian = input.to_vec();
                xdsp::from_polar(&mut cartesian);
                transform.inverse(&cartesian, output);
            } else {
                transform.inverse(input, output);
            }
        }
    }
}

/// Computes the command and completes it at once.
pub extern "system" fn XDspSubmitCommand(
    stream: *mut Stream,
    command: *const Command,
    ticket: *mut u32,
) -> HRESULT {
    if stream.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a stream from `XDspActivate`, the title's command.
    let (stream, command) = unsafe { (&mut *stream, &*command) };
    if !command.output.is_null() {
        let samples = (stream.frames * stream.channels) as usize;
        // SAFETY: the title's output buffer holds one block of the stream.
        let output = unsafe { std::slice::from_raw_parts_mut(command.output, samples) };
        if command.input.is_null() {
            output.fill(0.0);
        } else {
            // SAFETY: the title's input buffer holds one block of the stream.
            let input = unsafe { std::slice::from_raw_parts(command.input, samples) };
            run(stream, command, input, output);
        }
    }
    let status = &mut *stream.status;
    let next = status.submitted.wrapping_add(1);
    // SAFETY: the status block is read by title code on other threads.
    unsafe {
        std::ptr::write_volatile(&mut status.submitted, next);
        std::ptr::write_volatile(&mut status.result, S_OK);
        std::ptr::write_volatile(&mut status.completed, next);
    }
    if !ticket.is_null() {
        // SAFETY: optional out parameter.
        unsafe { ticket.write(next) };
    }
    S_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dsp",
        dlls: &["chad.dll"],
        exports: vec![
            hle_export!(XDspConnectWithMaximumStreamLimit),
            hle_export!(XDspDisconnect),
            hle_export!(XDspActivate),
            hle_export!(XDspDeactivate),
            hle_export!(XDspSubmitCommand),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submit(
        stream: *mut Stream,
        input: &[f32],
        output: &mut [f32],
        multiplier: *const f32,
    ) -> u32 {
        let command = Command {
            begin_scale: 1.0,
            end_scale: 1.0,
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            multiplier,
        };
        let mut ticket = 0;
        assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);
        ticket
    }

    #[test]
    fn convolution_stream_as_the_title_drives_it() {
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                24,
                0x80,
                1,
                &mut dsp
            ),
            S_OK
        );
        // A response that delays by 3 frames at half gain, in one partition.
        let mut taps = vec![0f32; 0x400];
        taps[3] = 0.5;
        let ir = xdsp::partition(&taps, 0x400);
        assert_eq!(ir.len(), 0x800);
        // Stereo, deinterleaved, one mono response, as the title asks for.
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: DEINTERLEAVED,
            frames: 0x400,
            channels: 2,
            ir_floats: 0x800,
            ir: ir.as_ptr(),
        };
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);
        let input: Vec<f32> = (0..0x800).map(|i| (i % 7) as f32).collect();
        let mut output = vec![9f32; 0x800];
        let ticket = submit(stream, &input, &mut output, std::ptr::null());
        // SAFETY: the stream's status block.
        let (completed, result) = unsafe { ((*status).completed, (*status).result) };
        assert_eq!((completed, result), (ticket, S_OK));
        for (c, (i, o)) in input.chunks(0x400).zip(output.chunks(0x400)).enumerate() {
            for n in 0..0x400 {
                let expected = if n >= 3 { 0.5 * i[n - 3] } else { 0.0 };
                assert!((o[n] - expected).abs() < 1e-3, "channel {c} frame {n}");
            }
        }
        assert_eq!(XDspDeactivate(stream), S_OK);

        // Mono streams refuse stereo options; transforms are mono only.
        let bad = StreamDesc {
            channels: 1,
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &bad, &mut status, &mut stream),
            E_INVALIDARG
        );
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }

    #[test]
    fn transform_streams_round_trip() {
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                0,
                4,
                1,
                &mut dsp
            ),
            S_OK
        );
        let mut streams = [std::ptr::null_mut(); 2];
        let mut status = std::ptr::null();
        for (stream, kind) in streams.iter_mut().zip([FORWARD_FFT, INVERSE_FFT]) {
            let desc = StreamDesc {
                kind,
                options: POLAR,
                frames: 0x200,
                channels: 1,
                ir_floats: 0,
                ir: std::ptr::null(),
            };
            assert_eq!(XDspActivate(dsp, &desc, &mut status, stream), S_OK);
        }
        let input: Vec<f32> = (0..0x200).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
        let window = vec![0.5f32; 0x200];
        let (mut spectrum, mut back) = (vec![0f32; 0x200], vec![0f32; 0x200]);
        submit(streams[0], &input, &mut spectrum, window.as_ptr());
        submit(streams[1], &spectrum, &mut back, std::ptr::null());
        for (b, i) in back.iter().zip(&input) {
            assert!((b - 0.5 * i).abs() < 1e-3, "{b} vs {}", 0.5 * i);
        }
        for stream in streams {
            assert_eq!(XDspDeactivate(stream), S_OK);
        }
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }
}
