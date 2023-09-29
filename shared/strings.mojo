alias CharPtr = Pointer[UInt8]


@always_inline
fn len(ptr: CharPtr) -> Int:
    """Get the length of a string."""
    return external_call["strlen", Int](ptr)


@always_inline
fn startswith(haystack: CharPtr, needle: CharPtr) -> Bool:
    """Check if a string starts with a prefix."""
    return (
        external_call["strncmp", Int, CharPtr, CharPtr](needle, haystack, len(needle))
        == 0
    )


@always_inline
fn startswith(haystack: String, needle: String) -> Bool:
    return startswith(to_ptr(haystack), to_ptr(needle))


fn prints(ptr: CharPtr, end: String = "\n") -> None:
    for i in range(0, len(ptr)):
        print_no_newline(chr(ptr[i].to_int()))
    print_no_newline(end)


fn prints(*ptrs: CharPtr) -> None:
    let ptr_list = VariadicList(ptrs)
    for i in range(ptr_list.__len__() - 1):
        prints(ptr_list[i], end=", ")
    prints(ptr_list[ptr_list.__len__() - 1])


fn split(buf: CharPtr, delim: CharPtr) -> Pointer[CharPtr]:
    var output = DynamicVector[CharPtr](1)
    var token_ptr: CharPtr
    token_ptr = external_call["strtok", CharPtr, CharPtr, CharPtr](buf, delim)
    if token_ptr == token_ptr.get_null():
        output.push_back(buf)
        return output.data
    while True:
        output.push_back(token_ptr)
        token_ptr = external_call["strtok", CharPtr, CharPtr, CharPtr](
            buf.get_null(), delim
        )
        if token_ptr == token_ptr.get_null():
            break
    let trimmed = Pointer[CharPtr].alloc(output.size)
    memcpy[CharPtr](trimmed, output.data, output.size)
    return trimmed


fn split(buf: CharPtr, delim: String) -> Pointer[CharPtr]:
    return split(buf, to_ptr(delim))


fn to_ptr(s: String) -> CharPtr:
    """Convert a string to a char_pointer."""
    let n = s.__len__()
    let ptr = CharPtr().alloc(n + 1)
    for i in range(n):
        ptr.store(i, ord(s[i]))
    ptr.store(n, ord("\0"))
    return ptr


@always_inline
fn strtol(ptr: CharPtr) -> Int:
    let endptr = Pointer[CharPtr].alloc(0)
    return external_call["strtol", Int, CharPtr, Pointer[CharPtr], Int](ptr, endptr, 10)


@always_inline
fn atol(ptr: CharPtr) -> Int:
    return external_call["atol", Int, CharPtr](ptr)


fn from_ptr(ptr: CharPtr) -> String:
    let out_ptr = Pointer[Int8].alloc(len(ptr))
    memcpy[Int8](out_ptr, ptr.bitcast[Int8](), len(ptr))
    return String(out_ptr, len(ptr))
