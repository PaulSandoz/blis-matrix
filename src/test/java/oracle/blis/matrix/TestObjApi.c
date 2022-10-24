/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

/*
cc -I ${BLIS_HOME}/include/${BLIS_ARCH}/ \
    ${BLIS_HOME}/lib/${BLIS_ARCH}/libblis.a \
    src/test/java/oracle/blis/matrix/TestObjApi.c
*/
#include <stdio.h>
#include "blis.h"

int main( int argc, char** argv )
{
	num_t dt;
	dim_t m, n, k;
	inc_t rs, cs;

	obj_t a;
	obj_t b;
	obj_t c;
	obj_t* alpha;
	obj_t* beta;


	// Create some matrix operands to work with.
	dt = BLIS_DOUBLE;
	m = 4; n = 5; k = 3; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &c );
	bli_obj_create( dt, m, k, rs, cs, &a );
	bli_obj_create( dt, k, n, rs, cs, &b );

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

	// Initialize the matrix operands.
	bli_randm( &a );
	bli_setm( &BLIS_ONE, &b );
	bli_setm( &BLIS_ZERO, &c );

	bli_printm( "a: randomized", &a, "%5.2f", "" );
	bli_printm( "b: set to 1.0", &b, "%5.2f", "" );
	bli_printm( "c: initial value", &c, "%5.2f", "" );

	// c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
	bli_gemm( alpha, &a, &b, beta, &c );

	bli_printm( "c: after gemm", &c, "%5.2f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );

	return 0;
}