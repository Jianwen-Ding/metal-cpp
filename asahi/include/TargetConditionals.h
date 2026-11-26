// metal-cpp on Asahi Linux presents itself as macOS to metal-cpp's platform
// checks (they only select which Metal entry points exist).
#pragma once
#ifndef TARGET_OS_MAC
#define TARGET_OS_MAC 1
#define TARGET_OS_OSX 1
#define TARGET_OS_IPHONE 0
#define TARGET_OS_IOS 0
#define TARGET_OS_VISION 0
#define TARGET_OS_LINUX 1
#endif
#ifndef __MAC_OS_X_VERSION_MIN_REQUIRED
#define __MAC_OS_X_VERSION_MIN_REQUIRED 140000
#endif
