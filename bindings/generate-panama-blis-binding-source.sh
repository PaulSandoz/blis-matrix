#!/bin/bash

blisHome=${BLIS_HOME}
blisArch=${BLIS_ARCH}

if [ ! -z "$1" ]; then
  blisHome=$1
fi

if [ ! -z "$2" ]; then
  blisArch=$2
fi

blisHeader=${blisHome}/include/${blisArch}/blis.h

echo Generating Panama blis source bindings for $blisHeader.

packageName=oracle.blis.binding

mkdir -p target

jextract \
   -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include \
   --dump-includes target/blis.includes.conf \
  $blisHeader

grep -e blis.h -e pthread target/blis.includes.conf > target/blis.includes.filtered.conf

jextract \
   -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include \
   --output target/generated-sources/blis \
   -l blis \
   --use-system-load-library \
   -t ${packageName} \
   @target/blis.includes.filtered.conf \
  $blisHeader

