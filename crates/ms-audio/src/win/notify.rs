//! `IMMNotificationClient`: forwards endpoint changes to the service thread.
//!
//! Callbacks run on an MMDevAPI thread and must not block: they only send on an
//! unbounded channel.

use super::Cmd;
use crossbeam_channel::Sender;
use windows::core::{implement, PCWSTR};
use windows::Win32::Foundation::PROPERTYKEY;
use windows::Win32::Media::Audio::{
    eConsole, eRender, EDataFlow, ERole, IMMDeviceEnumerator, IMMNotificationClient,
    IMMNotificationClient_Impl, MMDeviceEnumerator, DEVICE_STATE, DEVICE_STATE_ACTIVE,
};
use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_ALL};

/// Creates the endpoint enumerator (COM must be initialised on this thread).
pub(super) fn create_enumerator() -> windows::core::Result<IMMDeviceEnumerator> {
    // SAFETY: standard COM activation.
    unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
}

/// A registered notification client; unregisters on drop.
pub(super) struct Notifier {
    enumerator: IMMDeviceEnumerator,
    client: IMMNotificationClient,
}

impl Notifier {
    pub(super) fn register(
        enumerator: &IMMDeviceEnumerator,
        tx: Sender<Cmd>,
    ) -> windows::core::Result<Self> {
        let client: IMMNotificationClient = Client { tx }.into();
        // SAFETY: `client` is a valid COM object; it stays alive in `self` until unregistered.
        unsafe { enumerator.RegisterEndpointNotificationCallback(&client)? };
        Ok(Self {
            enumerator: enumerator.clone(),
            client,
        })
    }
}

impl Drop for Notifier {
    fn drop(&mut self) {
        // SAFETY: unregisters the client registered in `register`.
        if let Err(e) = unsafe {
            self.enumerator
                .UnregisterEndpointNotificationCallback(&self.client)
        } {
            tracing::warn!("ms-audio: cannot unregister device notifications: {e}");
        }
    }
}

#[implement(IMMNotificationClient)]
struct Client {
    tx: Sender<Cmd>,
}

fn id_string(id: &PCWSTR) -> Option<String> {
    if id.is_null() {
        return None;
    }
    // SAFETY: MMDevAPI passes a valid NUL-terminated string for the call's duration.
    unsafe { id.to_string().ok() }
}

impl IMMNotificationClient_Impl for Client_Impl {
    fn OnDeviceStateChanged(
        &self,
        pwstrdeviceid: &PCWSTR,
        dwnewstate: DEVICE_STATE,
    ) -> windows::core::Result<()> {
        if let Some(id) = id_string(pwstrdeviceid) {
            let _ = self.tx.send(Cmd::DeviceState {
                id,
                active: dwnewstate == DEVICE_STATE_ACTIVE,
            });
        }
        Ok(())
    }

    fn OnDeviceAdded(&self, _pwstrdeviceid: &PCWSTR) -> windows::core::Result<()> {
        Ok(())
    }

    fn OnDeviceRemoved(&self, pwstrdeviceid: &PCWSTR) -> windows::core::Result<()> {
        if let Some(id) = id_string(pwstrdeviceid) {
            let _ = self.tx.send(Cmd::DeviceState { id, active: false });
        }
        Ok(())
    }

    fn OnDefaultDeviceChanged(
        &self,
        flow: EDataFlow,
        role: ERole,
        pwstrdefaultdeviceid: &PCWSTR,
    ) -> windows::core::Result<()> {
        if flow == eRender && role == eConsole {
            let _ = self
                .tx
                .send(Cmd::DefaultChanged(id_string(pwstrdefaultdeviceid)));
        }
        Ok(())
    }

    fn OnPropertyValueChanged(
        &self,
        _pwstrdeviceid: &PCWSTR,
        _key: &PROPERTYKEY,
    ) -> windows::core::Result<()> {
        Ok(())
    }
}
