use ms_analysis::Frame;

/// Speaker position of one interleaved channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    FL,
    FR,
    FC,
    LFE,
    BL,
    BR,
    SL,
    SR,
    Other,
}

const K: f32 = 0.70710677;

/// Default channel layout for a channel count.
pub fn layout_for(channels: u16) -> Vec<Channel> {
    use Channel::*;
    match channels {
        0 => Vec::new(),
        1 => vec![FC],
        2 => vec![FL, FR],
        6 => vec![FL, FR, FC, LFE, SL, SR],
        8 => vec![FL, FR, FC, LFE, BL, BR, SL, SR],
        n => {
            let mut v = vec![FL, FR];
            v.resize(n as usize, Other);
            v
        }
    }
}

/// Downmixes interleaved samples to stereo. Real-time safe: no allocation.
///
/// Processes `min(interleaved.len() / layout.len(), out.len())` frames;
/// a trailing partial frame is ignored and an empty layout writes nothing.
pub fn downmix(interleaved: &[f32], layout: &[Channel], out: &mut [Frame]) {
    if layout.is_empty() {
        return;
    }
    let mono = layout == [Channel::FC];
    for (chunk, o) in interleaved.chunks_exact(layout.len()).zip(out.iter_mut()) {
        if mono {
            *o = [chunk[0], chunk[0]];
            continue;
        }
        let (mut l, mut r) = (0.0f32, 0.0f32);
        for (&s, ch) in chunk.iter().zip(layout) {
            match ch {
                Channel::FL => l += s,
                Channel::FR => r += s,
                Channel::FC | Channel::LFE => {
                    l += s * K;
                    r += s * K;
                }
                Channel::BL | Channel::SL => l += s * K,
                Channel::BR | Channel::SR => r += s * K,
                Channel::Other => {}
            }
        }
        *o = [l, r];
    }
}
