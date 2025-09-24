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
  --dump-includes target/blis.includes.conf \
  $blisHeader

# Filter jextract directives to reduce what is generated
grep \
  -e "--include-constant BLIS_" \
  -e "--include-var BLIS_" \
  -e "--include-function bli_" \
  -e " obj_" \
  -e " dcomplex" \
  target/blis.includes.conf > target/blis.includes.filtered.conf
grep -Ev "thread|BLIS_SINGLE_COMM" \
  target/blis.includes.filtered.conf > target/blis.includes.filtered.conf2

jextract \
  --output target/generated-sources/blis \
  -l blis \
  --use-system-load-library \
  -t ${packageName} \
  @target/blis.includes.filtered.conf2 \
  $blisHeader

