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

echo Generating Panama blis bindings for $blisHeader

groupId=oracle.blis.matrix
artifactId=blis-binding
version=1.0-SNAPSHOT

packageName=${groupId}.binding

if [ ! -d target/blis/classes ]; then
  mkdir -p target

  # Extract symbols
  jextract \
     -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include \
     --dump-includes target/blis.includes.conf \
    $blisHeader

  # Filter to only symbols from blis header
  grep blis.h target/blis.includes.conf > target/blis.includes.filtered.conf

  jextract \
     -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include \
     --output target/blis/classes \
     -l blis -t ${packageName} \
     @target/blis.includes.filtered.conf \
    $blisHeader
fi

blisJarFile=${artifactId}-${version}.jar
jar --create --file target/${blisJarFile} -C target/blis/classes/ .

mvn install:install-file -Dfile=target/${blisJarFile} \
  -DgroupId=${groupId} \
  -DartifactId=${artifactId} \
  -Dversion=${version} \
  -Dpackaging=jar \
  -DgeneratePom=true

echo Installed BLIS Java bindings as maven artifact ${groupId}:${artifactId}:${version}
