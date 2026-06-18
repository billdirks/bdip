# GitHub Release Guide for bdip

Follow these steps to compile, package, and distribute both your GUI app and CLI tool on GitHub Releases.

> [!TIP]
> You only need to configure `cargo-bundle` once. After the initial setup, you just repeat the "Compile & Package" steps for future releases!

## 1. Initial Setup (One-time)

First, install the community tool `cargo-bundle` which automatically creates Mac `.app` bundles from Rust binaries.

```bash
cargo install cargo-bundle
```

Next, ensure this metadata block is at the very bottom of your `bdip/Cargo.toml` file so `cargo-bundle` knows how to package the app and where to find your icon.

```toml
[package.metadata.bundle]
name = "bdip"
identifier = "com.billdirks.bdip"
icon = ["bdip/assets/goat.icns"]
version = "<VERSION>"
copyright = "Copyright (c) 2026, William Dirks"
```

## 2. Compile & Package

### The UI Application (`bdip`)

> [!IMPORTANT]
> **Apple Developer Setup (One-Time):** 
> 1. Enroll in the [Apple Developer Program](https://developer.apple.com/programs/) ($99/year).
> 2. Create a "Developer ID Application" certificate in your Apple Developer account and install it in your Mac's Keychain.
> 3. Generate an "App-Specific Password" at [appleid.apple.com](https://appleid.apple.com) for the `notarytool`.

Generate the `.app` bundle:
```bash
cargo bundle --release -p bdip
```

Now, cryptographically sign the app using your Apple Developer certificate (replace with your actual name and Team ID from your Keychain).
```bash
codesign --force --deep --options runtime --timestamp --sign "Developer ID Application: William Dirks (<TEAM_ID>)" target/release/bundle/osx/bdip.app
```

Next, compress the signed app into a zip file:
```bash
ditto -c -k --keepParent target/release/bundle/osx/bdip.app bdip-mac-v<VERSION>.zip
```

Finally, upload it to Apple's automated Notary Service to get it cleared by Gatekeeper:
```bash
xcrun notarytool submit bdip-mac-v<VERSION>.zip --apple-id "<your-email@example.com>" --password "<app-specific-password>" --team-id "<TEAM_ID>" --wait
```

Once the notary tool says "Accepted", embed the offline ticket into the app and re-zip it:
```bash
xcrun stapler staple target/release/bundle/osx/bdip.app
ditto -c -k --keepParent target/release/bundle/osx/bdip.app bdip-mac-v<VERSION>.zip
```
*(Your new zip file is now offline-ready and officially cleared to open on any Mac without warnings!)*

### The CLI Tool (`bdip-cli`)
The CLI is distributed as a raw executable, so it just needs a standard release build:
```bash
cargo build --release -p bdip-cli
```

Sign the executable with your Apple Developer certificate:
```bash
codesign --force --options runtime --timestamp --sign "Developer ID Application: William Dirks (<TEAM_ID>)" target/release/bdip-cli
```

Finally, compress the signed executable into a zip file:
```bash
zip -j bdip-cli-mac-v<VERSION>.zip target/release/bdip-cli
```

Just like the GUI app, you must upload the CLI zip to the Notary service so Gatekeeper doesn't block it:
```bash
xcrun notarytool submit bdip-cli-mac-v<VERSION>.zip --apple-id "<your-email@example.com>" --password "<app-specific-password>" --team-id "<TEAM_ID>" --wait
```
*(Note: Unlike Mac `.app` bundles, Apple does not support "stapling" raw terminal executables. So there is no stapler step here. Once Notarytool says "Accepted", your zip file is immediately ready to publish!)*

## 3. Publish to GitHub

Now that you have your two `.zip` files, it's time to publish them!

1. Open your repository on GitHub.
2. Look at the right-hand sidebar and click **Releases**, then click **Draft a new release**.
3. In the **Choose a tag** dropdown, type `v<VERSION>` and click **Create new tag**.
4. Give your release a Title (e.g., "Release v<VERSION>") and add a description of any new features or bug fixes.
5. Under **Attach binaries by dropping them here**, drag and drop your two files:
   - `bdip-mac-v<VERSION>.zip`
   - `bdip-cli-mac-v<VERSION>.zip`
6. Click **Publish release**.
