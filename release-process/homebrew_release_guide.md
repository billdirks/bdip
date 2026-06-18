# Homebrew Release Guide for bdip

This guide explains how to create your own "Homebrew Tap" so users can easily install your applications using `brew install`. 

> [!IMPORTANT]
> **Prerequisite:** This guide assumes you have already followed the **GitHub Release Guide** and successfully published your `.zip` files to a GitHub Release (e.g., `v<VERSION>`). Homebrew does not host your files; it simply automates downloading them from your GitHub Releases!

## 1. Create Your Homebrew Tap Repository

Homebrew looks for installation scripts in GitHub repositories named with a `homebrew-` prefix.

1. Go to GitHub and create a new, public repository named exactly: **`homebrew-bdip`**
2. Clone this empty repository to your local machine:
   ```bash
   git clone https://github.com/billdirks/homebrew-bdip.git
   cd homebrew-bdip
   ```
3. Inside that folder, create two directories:
   ```bash
   mkdir Formula
   mkdir Casks
   ```

## 2. Get Your Checksums

Homebrew requires the SHA-256 hash of your `.zip` files to ensure they haven't been tampered with. Run this command on your local zip files (the ones you uploaded to GitHub):

```bash
shasum -a 256 path/to/bdip-mac-v<VERSION>.zip
shasum -a 256 path/to/bdip-cli-mac-v<VERSION>.zip
```
*Copy the two long hashes it outputs.*

## 3. Create the CLI Formula

Inside the `Formula` folder, create a file named `bdip-cli.rb`. This tells Homebrew how to install the raw executable.

```ruby
class BdipCli < Formula
  desc "High-performance command-line shader application"
  homepage "https://github.com/billdirks/bdip"
  version "<VERSION>"
  
  # The URL to the zip file you uploaded to GitHub Releases
  url "https://github.com/billdirks/bdip/releases/download/v#{version}/bdip-cli-mac-v#{version}.zip"
  
  sha256 "feee4cb63762ce6aad77b2b79370c5b2e9621aa288f0fdb94483f16ede55efc9"

  def install
    # Homebrew unzips the file automatically. 
    # This line moves the raw binary into the user's PATH.
    bin.install "bdip-cli"
  end

  def caveats
    <<~EOS
      Note: bdip also has a graphical user interface (GUI) application!
      If you are interested in interactive editing, you can install it by running:
        brew install --cask bdip
    EOS
  end
end
```

## 4. Create the GUI Cask

Inside the `Casks` folder, create a file named `bdip.rb`. This tells Homebrew how to install the macOS `.app` bundle.

```ruby
cask "bdip" do
  version "<VERSION>"
  
  sha256 "ab0964e953bba6dcea0a7a3dee10446bb48553f9a707cc9e01ecca27cc22ae5b"

  url "https://github.com/billdirks/bdip/releases/download/v#{version}/bdip-mac-v#{version}.zip"
  name "bdip"
  desc "High-performance desktop GUI application"
  homepage "https://github.com/billdirks/bdip"

  # This tells Homebrew to move the extracted .app to /Applications
  app "bdip.app"
  
  caveats do
    <<~EOS
      Note: bdip also has a powerful command-line interface (CLI) tool for batch processing!
      If you are interested, you can install it by running:
        brew install bdip-cli
    EOS
  end
end
```

## 5. Publish and Share!

Commit and push those two files to your `homebrew-bdip` repository:

```bash
git add .
git commit -m "Add bdip v<VERSION>"
git push origin main
```

**You are done!** Users can now install your applications by running:

```bash
# Add your custom tap to their local Homebrew
brew tap billdirks/bdip
brew trust billdirks/bdip

# Install the CLI tool
brew install bdip-cli

# Install the UI App
brew install --cask bdip
```

> [!TIP]
> **For future updates (e.g., `v<NEXT_VERSION>`):** You do not need to create a new repository. You just open these two `.rb` files, bump the `version` number, paste the new `sha256` hashes, and push to GitHub!
