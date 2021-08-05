
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

unsigned char *stbi_write_png_to_mem_wrapper(const unsigned char *pixels, int stride_bytes, int x, int y, int n, int *out_len) {
	return stbi_write_png_to_mem(pixels, stride_bytes, x, y, n, out_len);
}

void stbiw_free_wrapper(void* p) {
	STBIW_FREE(p);
}

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
