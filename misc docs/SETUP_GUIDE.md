# VideoGS iOS Viewer - Complete Setup Guide

This guide provides detailed step-by-step instructions to clone, configure, and run the VideoGS iOS Viewer for streaming volumetric videos on iOS devices.

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Directory Structure](#directory-structure)
- [Code Changes](#code-changes)
- [Building and Running](#building-and-running)
- [Troubleshooting](#troubleshooting)
- [Controls](#controls)

## Overview

**Project:** V³: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians  
**Original Repository:** https://github.com/zhangzhr4/VideoGS_IOS_viewers  
**This Fork:** https://github.com/Amartya1911/VideoGS_IOS_Viewer  
**Purpose:** iOS viewer for streaming volumetric video using 2D Dynamic Gaussians

## Prerequisites

### System Requirements
- macOS with Xcode 15.0 or later
- iOS device running iOS 16.0 or later
- Active Apple Developer account (for device deployment)
- Git installed on your system

### Required Downloads
1. **OpenCV2 Framework (iOS version) - v4.12.0**
   - Download from: https://opencv.org/releases/
   - Specific version: `opencv-4.12.0-ios-framework.zip` (~221MB compressed)
   - Direct link: https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-ios-framework.zip

2. **Video Checkpoint Data**
   - Download from: [SharePoint Link](https://5xmbb1-my.sharepoint.com/:f:/g/personal/auwang_5xmbb1_onmicrosoft_com/Eld7m8aca11Kloo3rZuwYzQBYwxOuAa9q9JLsoZdT_bKPA?e=Ntv8yk)
   - File: `ykx_boxing_long_qp15_380` (~638MB)
   - Contains 67 group folders with video data

## Installation Steps

### Step 1: Clone the Repository

```bash
# Navigate to your desired directory
cd ~/Desktop

# Clone the repository
git clone https://github.com/Amartya1911/VideoGS_IOS_Viewer.git rebuilt_iosViewer

# Navigate into the project
cd rebuilt_iosViewer
```

### Step 2: Install OpenCV2 Framework (v4.12.0)

```bash
# Download OpenCV 4.12.0 if not already downloaded
# wget https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-ios-framework.zip

# Unzip the downloaded file
unzip ~/Downloads/opencv-4.12.0-ios-framework.zip -d ~/Downloads/

# Copy to project root
cp -R ~/Downloads/opencv2.framework .

# Verify the copy
ls -la opencv2.framework/
```

**Expected structure:**
```
opencv2.framework/
├── Headers/
├── Modules/
├── Resources/
├── Versions/
└── opencv2 (binary)
```

**Important:** OpenCV2 framework is ~547MB and is intentionally excluded from git via `.gitignore`.

### Step 3: Install Video Data

```bash
# Copy the video data folder to MetalSplat directory
cp -R ~/Downloads/ykx_boxing_long_qp15_380 MetalSplat/

# Verify the copy
ls -la MetalSplat/ykx_boxing_long_qp15_380/
```

**Expected structure:**
```
MetalSplat/ykx_boxing_long_qp15_380/
├── group0/ (20 MP4 files)
├── group1/ (20 MP4 files)
├── group2/ (20 MP4 files)
├── ...
└── group65+/ (20 MP4 files)
```

**Important:** Video data is ~638MB and is intentionally excluded from git via `.gitignore`.

### Step 4: Verify Package Dependencies

This fork has already updated the Forge dependency. The project file should reference:

```xml
repositoryURL = "https://github.com/drunknbass/Forge";
```

If you're working with the original repo, you'll need to update this manually. See [Code Changes](#code-changes) section.

## Directory Structure

### Final Project Structure

```
rebuilt_iosViewer/
│
├── opencv2.framework/                   [547MB - External dependency, NOT in git]
│   ├── Headers/
│   ├── Modules/
│   ├── Resources/
│   ├── Versions/
│   └── opencv2
│
├── MetalSplat.xcodeproj/               [Xcode project]
│   ├── project.pbxproj                 [MODIFIED - Forge URL]
│   └── project.xcworkspace/
│
├── MetalSplat/                         [Main source folder]
│   │
│   ├── ykx_boxing_long_qp15_380/      [638MB - Video data, NOT in git]
│   │   ├── group0/ (20 MP4 files)
│   │   ├── group1/ (20 MP4 files)
│   │   ├── ...
│   │   └── group65+/ (20 MP4 files)
│   │
│   ├── group_info_ykx_380.json        [Video metadata]
│   ├── viewer_min_max_ykx_380.json    [Viewer configuration]
│   │
│   ├── AppDelegate.swift
│   ├── MainViewController.swift
│   ├── SplatCloud.swift
│   ├── SplatSimpleView.swift
│   ├── ARSplatView.swift
│   ├── SplatShaders.metal             [Compute shaders]
│   │
│   ├── Utils/
│   │   ├── splat_utils.mm             [Objective-C++ utilities]
│   │   ├── DragHelper.swift
│   │   ├── MetalBuffer.swift
│   │   ├── Metal+Extensions.swift
│   │   ├── ARBackgroundRenderer.swift
│   │   └── ARPerspectiveCamera.swift
│   │
│   ├── Models/
│   │   ├── Models.swift
│   │   ├── lego_60k.ply
│   │   └── mic_60k.ply
│   │
│   ├── Assets/
│   ├── Assets.xcassets/
│   ├── Main.storyboard
│   ├── LaunchScreen.storyboard
│   │
│   ├── BridgingHeader.h               [Swift-ObjC bridge]
│   ├── ShaderTypes.h
│   ├── OpencvTest.mm
│   ├── OpencvTest.h
│   └── Info.plist
│
├── assets/
│   └── demo.jpeg
│
├── README.md
├── SETUP_GUIDE.md                      [This file]
├── LICENSE.md
└── .gitignore                          [Excludes opencv2.framework and video data]
```

## Code Changes

### Modified Files

#### 1. `MetalSplat.xcodeproj/project.pbxproj`

**Change:** Updated Forge package dependency URL

**Location:** Line ~485 in the `XCRemoteSwiftPackageReference` section

**Before:**
```xml
424365322AB503E500C3EF55 /* XCRemoteSwiftPackageReference "Forge" */ = {
    isa = XCRemoteSwiftPackageReference;
    repositoryURL = "https://github.com/Hi-Rez/Forge";
    requirement = {
        kind = upToNextMajorVersion;
        minimumVersion = 1.4.2;
    };
};
```

**After:**
```xml
424365322AB503E500C3EF55 /* XCRemoteSwiftPackageReference "Forge" */ = {
    isa = XCRemoteSwiftPackageReference;
    repositoryURL = "https://github.com/drunknbass/Forge";
    requirement = {
        kind = upToNextMajorVersion;
        minimumVersion = 1.4.2;
    };
};
```

**Reason:** The original Hi-Rez/Forge repository is no longer available. The drunknbass fork is an active, maintained version.

**Manual Update (if needed):**
```bash
# Use sed to replace the URL in the project file
sed -i '' 's|https://github.com/Hi-Rez/Forge|https://github.com/drunknbass/Forge|g' MetalSplat.xcodeproj/project.pbxproj
```

### No Other Code Changes Required

All other source files remain unchanged from the original repository. The project uses:
- **Satin** (v13.0.0+) - 3D rendering framework
- **Forge** (v1.4.2+) - Metal utilities
- **SatinCore** - Core rendering components
- **OpenCV** (v4.12.0) - Computer vision library

## Building and Running

### Step 1: Open the Project

```bash
# From project directory
open MetalSplat.xcodeproj
```

Or double-click `MetalSplat.xcodeproj` in Finder.

### Step 2: Configure Xcode

1. **Select your development team:**
   - Click on the project in the navigator
   - Select the "MetalSplat" target
   - Go to "Signing & Capabilities"
   - Select your Apple Developer team

2. **Connect your iOS device:**
   - Connect via USB
   - Trust the computer on your device if prompted

3. **Select your device:**
   - In the Xcode toolbar, select your iOS device from the scheme dropdown

### Step 3: Install Dependencies

Xcode will automatically:
- Fetch Satin package from GitHub
- Fetch Forge package from the updated URL
- Link opencv2.framework

This may take a few minutes on first build.

### Step 4: Build and Run

1. Press `⌘R` or click the "Play" button
2. Xcode will compile and install the app
3. If prompted, trust the developer certificate on your device:
   - Settings → General → VPN & Device Management
   - Trust your developer certificate

## Troubleshooting

### Common Issues

#### 1. "Forge package not found"

**Problem:** Original repository URL is dead  
**Solution:** Ensure you've updated the Forge URL in `project.pbxproj` as described above

```bash
# Verify the change
grep -n "drunknbass/Forge" MetalSplat.xcodeproj/project.pbxproj
```

#### 2. "opencv2.framework not found"

**Problem:** Framework not in correct location  
**Solution:** Verify framework is in project root:

```bash
ls -la opencv2.framework/
# Should show: Headers, Modules, Resources, etc.
```

If missing, re-download OpenCV 4.12.0 and copy to project root.

#### 3. "Video data not loading"

**Problem:** Data folder in wrong location  
**Solution:** Ensure data is directly under MetalSplat:

```bash
ls -la MetalSplat/ykx_boxing_long_qp15_380/
# Should show: group0, group1, group2, etc.
```

**Correct:** `MetalSplat/ykx_boxing_long_qp15_380/`  
**Incorrect:** `MetalSplat/data/ykx_boxing_long_qp15_380/`

#### 4. Build fails with Metal errors

**Problem:** Deployment target mismatch  
**Solution:** 
- Minimum iOS version: 16.0
- Ensure device runs iOS 16.0+
- Check: Project Settings → Deployment Info → iOS Deployment Target

#### 5. Code signing errors

**Problem:** No development team selected  
**Solution:**
- Add Apple ID to Xcode: Preferences → Accounts
- Select team in project settings
- Or use automatic signing

#### 6. "OpenCV version mismatch"

**Problem:** Wrong OpenCV version installed  
**Solution:** Ensure you have OpenCV 4.12.0 specifically:
```bash
# Check version in framework info
cat opencv2.framework/Resources/Info.plist | grep -A 1 "CFBundleShortVersionString"
```

## Controls

Once the app is running:

### Basic Controls
- **Tap cover image** - Load volumetric video
- **Play button** - Start video playback
- **Pause button** - Pause playback
- **Slider** - Scrub to specific frame
- **Back button** - Return to cover screen

### Gestures
- **One finger drag** - Rotate the view
- **Two finger pinch** - Scale (zoom in/out)
- **Two finger drag** - Translate (pan)

## Streaming Version

For remote streaming:
1. Host the `ykx_boxing_long_qp15_380` folder on a web server
2. Modify the data path in the code to point to the server URL
3. Ensure network connectivity between device and server

## Performance Tips

- **First load:** May take 10-15 seconds to load video data
- **Smooth playback:** iPhone 12 or newer recommended
- **Storage:** App bundle + data = ~1.2GB
- **Network streaming:** Requires stable WiFi (>10 Mbps recommended)

## Additional Resources

- **Original Paper:** V³: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians
- **Original GitHub:** https://github.com/zhangzhr4/VideoGS_IOS_viewers
- **Report Issues:** https://github.com/Amartya1911/VideoGS_IOS_Viewer/issues
- **Metal Shaders:** See `SplatShaders.metal` for compute shader implementation

## Preserving Working State

After successfully building, create a git checkpoint:

```bash
# Commit all changes (excluding large files)
git add .
git commit -m "Working state: app running successfully"

# Create a tag for easy reference
git tag -a working-v1.0 -m "Working baseline with Forge fix and data installed"
git push origin main --tags
```

To restore this state later:
```bash
git checkout working-v1.0
# Then re-copy opencv2.framework and data folder from Downloads
```

## File Sizes Summary

| Item | Size | Location | In Git? |
|------|------|----------|---------|
| opencv2.framework | ~547MB | Project root | ❌ No |
| Video data (ykx_boxing_long_qp15_380) | ~638MB | MetalSplat/ | ❌ No |
| Project source code | ~5MB | MetalSplat/ | ✅ Yes |
| **Total (with data)** | **~1.2GB** | | |
| **Total (git clone)** | **~5MB** | | |

**Note:** Large files are excluded via `.gitignore` to keep repository size manageable. You must download and install them separately following Steps 2 and 3.

## License

See LICENSE.md in the project root.

## Credits

- Original implementation by Zhang et al.
- Metal rendering via Satin framework
- Forge utilities by drunknbass
- OpenCV for image processing (v4.12.0)

---

**Setup Guide Version:** 1.0  
**Last Updated:** November 23, 2025  
**Tested On:** Xcode 15.0, iOS 17.0, OpenCV 4.12.0
