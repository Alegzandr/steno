import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

type InputDevice = {
  name: string;
  isDefault: boolean;
};

// Empty string is the <select> value for "follow the system default", which
// maps to a null override in the backend config.
const SYSTEM_DEFAULT = "";

type Props = {
  // Recording opens the device, so the choice is frozen while one is running;
  // it takes effect on the next recording anyway.
  disabled: boolean;
};

/**
 * The microphone dropdown for the placeholder UI. Lists the input devices
 * present now and persists the choice. Folds into real settings in phase 5.
 */
export function DevicePicker({ disabled }: Props) {
  const [devices, setDevices] = useState<InputDevice[]>([]);
  const [saved, setSaved] = useState<string>(SYSTEM_DEFAULT);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      invoke<InputDevice[]>("enumerate_input_devices"),
      invoke<string | null>("input_device"),
    ])
      .then(([list, override]) => {
        setDevices(list);
        setSaved(override ?? SYSTEM_DEFAULT);
      })
      .catch((cause) => setError(String(cause)));
  }, []);

  function choose(name: string) {
    setSaved(name);
    setError(null);
    invoke("set_input_device", {
      name: name === SYSTEM_DEFAULT ? null : name,
    }).catch((cause) => setError(String(cause)));
  }

  // The saved device may have been unplugged since it was chosen. The backend
  // silently falls back to the default in that case; surface it here so the
  // dropdown showing "System default" is not a lie.
  const missing =
    saved !== SYSTEM_DEFAULT && !devices.some((device) => device.name === saved);

  return (
    <div className="device-picker">
      <label className="device-picker-label" htmlFor="input-device">
        Microphone
      </label>
      <select
        id="input-device"
        className="device-picker-select"
        value={missing ? SYSTEM_DEFAULT : saved}
        disabled={disabled}
        onChange={(event) => choose(event.target.value)}
      >
        <option value={SYSTEM_DEFAULT}>System default</option>
        {devices.map((device) => (
          <option key={device.name} value={device.name}>
            {device.name}
            {device.isDefault ? " (default)" : ""}
          </option>
        ))}
      </select>
      {missing && (
        <p className="device-picker-note">
          “{saved}” isn’t connected — using the system default.
        </p>
      )}
      {error && <p className="device-picker-note">{error}</p>}
    </div>
  );
}
