//! Bluetooth endpoint detection via `PKEY_Device_EnumeratorName`.

use std::iter::once;
use tracing::debug;
use windows::core::PCWSTR;
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_EnumeratorName;
use windows::Win32::Media::Audio::IMMDeviceEnumerator;
use windows::Win32::System::Com::StructuredStorage::{PropVariantClear, PropVariantToBSTR};
use windows::Win32::System::Com::STGM_READ;

/// True if the endpoint `endpoint_id` sits on a Bluetooth bus. Any failure → false.
pub(super) fn is_bluetooth(enumerator: Option<&IMMDeviceEnumerator>, endpoint_id: &str) -> bool {
    let Some(enumerator) = enumerator else {
        return false;
    };
    match enumerator_name(enumerator, endpoint_id) {
        Ok(name) => is_bluetooth_enumerator(&name),
        Err(e) => {
            debug!("ms-audio: cannot read enumerator name of {endpoint_id}: {e}");
            false
        }
    }
}

/// `BTHENUM` (classic) or `BTHLEDEVICE` (LE audio), case-insensitive.
fn is_bluetooth_enumerator(name: &str) -> bool {
    name.eq_ignore_ascii_case("BTHENUM") || name.eq_ignore_ascii_case("BTHLEDEVICE")
}

fn enumerator_name(
    enumerator: &IMMDeviceEnumerator,
    endpoint_id: &str,
) -> windows::core::Result<String> {
    let wide: Vec<u16> = endpoint_id.encode_utf16().chain(once(0)).collect();
    // SAFETY: `wide` is NUL-terminated and outlives the call; the PROPVARIANT is cleared
    // after conversion.
    unsafe {
        let device = enumerator.GetDevice(PCWSTR(wide.as_ptr()))?;
        let store = device.OpenPropertyStore(STGM_READ)?;
        let mut value = store.GetValue(&PKEY_Device_EnumeratorName)?;
        let text = PropVariantToBSTR(&value);
        let _ = PropVariantClear(&mut value);
        Ok(text?.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::is_bluetooth_enumerator;

    #[test]
    fn enumerator_names() {
        assert!(is_bluetooth_enumerator("BTHENUM"));
        assert!(is_bluetooth_enumerator("bthenum"));
        assert!(is_bluetooth_enumerator("BthLEDevice"));
        assert!(!is_bluetooth_enumerator("USB"));
        assert!(!is_bluetooth_enumerator("HDAUDIO"));
        assert!(!is_bluetooth_enumerator(""));
        assert!(!is_bluetooth_enumerator("BTHENUMX"));
    }
}
