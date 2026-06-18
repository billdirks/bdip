# Cargo Release Guide for bdip

This guide outlines the steps required to publish the three Rust crates in this workspace to [crates.io](https://crates.io/).

> [!IMPORTANT]
> Because `bdip` and `bdip-cli` both depend on `bdip-core`, you **must** publish `bdip-core` first! Cargo requires all dependencies to be available on crates.io before dependent packages can be published.

## 1. Preparation

Before publishing, ensure you are logged into crates.io via the Cargo CLI. If you haven't done this before, go to crates.io, generate an API token in your Account Settings, and run:
```bash
cargo login <your-api-token>
```

Also, ensure that:
1. You have cleanly committed all your changes to Git.
2. The `version` string in all three `Cargo.toml` files has been updated.
3. The `bdip-core` version requirement in both `bdip/Cargo.toml` and `bdip-cli/Cargo.toml` matches the new version you are releasing.

## 2. Publish `bdip-core`

Since the other crates rely on `bdip-core`, it must be published first.

```bash
cd bdip-core
cargo publish
cd ..
```

*Note: After publishing, it may take a few seconds for crates.io's index to update. If the next step fails saying the new version of `bdip-core` cannot be found, simply wait 10-20 seconds and try again.*

## 3. Publish `bdip-cli`

Once `bdip-core` is successfully published and the index is updated, you can publish the command-line tool.

```bash
cd bdip-cli
cargo publish
cd ..
```

## 4. Publish `bdip`

Finally, publish the desktop GUI application.

```bash
cd bdip
cargo publish
cd ..
```

> [!TIP]
> If you want to verify that the crates will package correctly without actually uploading them, you can run `cargo publish --dry-run` inside any of the crate directories.

## 5. Verify the Release

Head over to [crates.io](https://crates.io/) and search for your crates to ensure the new versions are live! Your users can now install your CLI tool natively via Cargo by running:
```bash
cargo install bdip-cli
```
