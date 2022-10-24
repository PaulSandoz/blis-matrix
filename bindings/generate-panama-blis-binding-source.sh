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

groupId=oracle.blis.matrix
artifactId=blis-binding
version=1.0-SNAPSHOT

packageName=${groupId}.binding

mkdir -p target

jextract \
   -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include \
   --dump-includes target/blis.includes.conf \
  $blisHeader

grep blis.h target/blis.includes.conf > target/blis.includes.filtered.conf

jextract \
   -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include \
   --output target/blis/sources \
   -l blis -t ${packageName} \
   --source \
   @target/blis.includes.filtered.conf \
  $blisHeader

blisJarFile=${artifactId}-${version}-sources.jar
jar --create --file target/${blisJarFile} -C target/blis/sources/ .

mvn install:install-file -Dfile=target/${blisJarFile} \
  -DgroupId=${groupId} \
  -DartifactId=${artifactId} \
  -Dversion=${version} \
  -Dpackaging=jar \
  -Dclassifier=sources

