# BLIS Matrix API

An example project that wraps the BLIS native library in Java using a Matrix
API.

The implementation of the Matrix API uses Panama's Foreign Function and Memory
(FFM) API and tooling to bind to the BLIS native library.

The result is a simple API and implementation. There are no memory copies
required to pass matrices from Java to BLIS native methods. There are no size
limits on the matrices, as would be the case if primitive arrays or ByteBuffers
were used. Memory management of the matrices can be explicitly controlled.

We get the efficiency of the BLIS native library with the productivity of Java.

## BLIS native library

[BLIS][BLIS] is a high performance CPU-based library for dense linear algebra
operations.

[BLIS]:[https://github.com/flame/blis

It provides a significant superset of the level 1-3 Basic Linear Algebra
Subprograms (BLAS). Especially noted is the level 3 performance e.g. GEneric
Matrix Multiplication (GEMM). It is one of only 2 libraries to offer GEMM-like
extensibility.

BLIS is developed by The Science of High Performance Computing Group at the
University of Texas at Austin

## Building the project

To build the project follow the instructions in subsections.

### Clone and build BLIS

BLIS may be cloned from here:

    https://github.com/flame/blis

Once cloned, configure and build:

```shell
./configure --enable-threading=pthreads auto
make -j
make install
```

On a machine with an Intel Haswell chip the blis header file will be located at
`include/haswell/blis.h` and the libraries will be located under `lib/haswell/`,
where `haswell` is the target architecture.

Set the environment variable `BLIS_HOME` to point to the local BLIS repository.

Set the environment variable `BLIS_ARCH` to be the target architecture. There by
`${BLIS_HOME}/include/${BLIS_ARCH}/blis.h` will be the path to the `blis.h`
header file.

### Download JDK 22

The Oracle builds of JDK 22 can be downloaded here:

    https://www.oracle.com/java/technologies/downloads/

Alternatively the OpenJDK build of JDK 22 can be downloaded here:

    https://jdk.java.net/22/

### Download and unpack the Panama tool jextract

`jextract` for JDK 22 can be downloaded from here:

    https://jdk.java.net/jextract/

Ensure that the jextract `bin` directory is added to the executable path, but be
careful to ensure the executable path of JDK 22 takes precedence.

### Generate Java source binding to the BLIS native library

From the `bindings` directory use maven to execute jextract to generate 
Java sources binding to the native BLIS library, compile them, and install 
them:

    mvn install

The `pom.xml` of the matrix project depends on the installed blis binding 
artefact:

```xml

<dependency>
    <groupId>oracle.blis</groupId>
    <artifactId>blis-binding</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

### Compile the source

Compile the source using maven, such as:

```shell
mvn package
```
