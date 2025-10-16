import oracle.blis.binding.obj_t;

import static oracle.blis.binding.blis_h.*;

void main() {
    try (Arena arena = Arena.ofConfined()) {
        MemorySegment a = obj_t.allocate(arena);
        bli_obj_create(BLIS_DOUBLE(), 4, 3, 0, 0, a);

        bli_randm(a);

        bli_printm(arena.allocateFrom("a: randomized"), a,
                arena.allocateFrom("%5.2f"),
                arena.allocateFrom(""));

        bli_obj_free(a);
    }
}