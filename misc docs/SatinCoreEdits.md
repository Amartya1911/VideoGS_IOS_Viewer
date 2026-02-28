# SatinCore Fixes for iOS Build

To compile the `Satin` dependency successfully with modern Xcode/Clang settings (which enforce C++11 narrowing rules), we had to manually patch the source code in the `DerivedData` checkouts.

## 1. The Issue
The build failed with **"non-constant-expression cannot be narrowed from type 'int' to 'uint32_t' in initializer list"**. This occurs when `int` values are used to initialize a struct expecting `uint32_t` (like `TriangleIndices` or `QuadIndices`) inside curly braces `{}`.

## 2. The Fixes

### A. Modified `Sources/SatinCore/Generators.mm`
We added explicit `(uint32_t)` casts to variables inside structure initializers.

**Example Change:**
```objective-c
// Before
geometry.indexData.push_back({ i, j, k });

// After
geometry.indexData.push_back({ (uint32_t)i, (uint32_t)j, (uint32_t)k });
```
*Note: This pattern was applied to multiple locations involving `TriangleIndices` and `QuadIndices`.*

### B. Modified `Sources/SatinCore/Triangulator.mm`
Similar casts were applied to triangulation logic.

**Example Change:**
```objective-c
// Before
geometry.indexData.push_back({ i0, i1, i2 });

// After
geometry.indexData.push_back({ (uint32_t)i0, (uint32_t)i1, (uint32_t)i2 });
```

---

## 3. Automated Reproduction Prompt

If you need to fix this again in a fresh environment without context, use the following prompt:

> **Prompt:**
> "I am trying to build an iOS project that depends on the 'Satin' library. The build is failing with C++ errors in `SatinCore` related to narrowing conversions (e.g., `non-constant-expression cannot be narrowed from type 'int' to 'uint32_t'`).
>
> Please find the `Satin` checkout in `DerivedData` (usually under `SourcePackages/checkouts/Satin`) and apply the following fixes to `Generators.mm` and `Triangulator.mm`:
>
> 1.  Search for brace initializers for index data, such as `{ i, j, k }`, `{ a, b, c, d }`, or similar patterns where integers are being pushed into an `indexData` vector.
> 2.  Cast all integer elements to `(uint32_t)` to satisfy the C++ compiler. 
>    - Change `{ i, j, k }` to `{ (uint32_t)i, (uint32_t)j, (uint32_t)k }`.
>    - Change `{ a, b, c, d }` to `{ (uint32_t)a, (uint32_t)b, (uint32_t)c, (uint32_t)d }`.
>
> Please use `sed` or file editing tools to apply these casts purely to the problematic lines."
