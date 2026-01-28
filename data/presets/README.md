# Presets Directory

This folder contains preset profiles for custom score configurations.

## Files

All files follow the naming pattern: `user_settings_example_*.json`

## Format

```json
{
  "_preset": "Profile Name",
  "_description": "Description of this profile",
  "custom_weights": {
    "indicator_key": weight,
    ...
  }
}
```

## Available Presets

- `user_settings_example_forensenbeleid.json` - Commuter profile (OV + Green)
- `user_settings_example_gezin.json` - Family with children
- `user_settings_example_landelijk.json` - Rural living (Space + Peace)
- `user_settings_example_stedelijk.json` - Urban & Wealthy
- `user_settings_example_betaalbaar.json` - Affordable housing

## Creating New Presets

1. Copy an existing preset
2. Modify `_preset`, `_description`, and `custom_weights`
3. Save with pattern `user_settings_example_yourname.json`
4. Restart dashboard - preset appears automatically!

## Notes

- Files starting with `_` in keys are ignored (metadata)
- Only `custom_weights` dictionary is used for loading
- Presets are loaded dynamically on dashboard start
