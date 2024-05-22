import pygame
import sys
import numpy as np
import numba
import math
import time
from numba import jit


'''
Copyright by splinesen.
GitHub: splinesen

Runs at 85~ FPS at 1080p on my high end PC.
'''


# Set canvas dimensions (same as in ShaderToy).
canvas_width = 1920
canvas_height = 1080

# Initialize pygame
pygame.init()

# Create the canvas
canvas = pygame.display.set_mode((canvas_width, canvas_height))
pygame.display.set_caption("UV Shader Canvas")

# Settings.
minIntensity = 1.5
maxIntensity = 5.5
combinedIntensity = 150.0
waveZoom = 4.0
waveStretch = 1.5

@jit(nopython=True, parallel=True)
def generate_uv_shader(canvas_width, canvas_height, iTime):
    pixels = np.zeros((canvas_width, canvas_height, 3), dtype=np.uint8)

    sin_iTime = math.sin(iTime * 0.33)
    cos_iTime = math.cos(iTime * 0.33)
    for y in numba.prange(canvas_height):
        # Map y-coordinate to OpenGL's coordinate system
        ogl_y = (canvas_height - y) - 1

        for x in range(canvas_width):
            # uv
            u = (((x + 0.5) * 2.0) - canvas_width) / canvas_height
            v = (((y + 0.5) * 2.0) - canvas_height) / canvas_height

            # uv0
            u0 = waveZoom * u
            v0 = waveZoom * v

            v0 += waveStretch * math.sin(u0 - (iTime * 0.75))

            lineIntensity = minIntensity + (maxIntensity * abs((u + iTime) % 2.0 - 1.0))

            glowWidth = abs(lineIntensity / (combinedIntensity * v0))

            r = glowWidth * (1.0 + sin_iTime)
            g = glowWidth * (1.0 - sin_iTime)
            b = glowWidth * (1.0 - cos_iTime)

            # Convert to RGB values in the range [0, 255]
            r = int(min(max(r * 255, 0), 255))
            g = int(min(max(g * 255, 0), 255))
            b = int(min(max(b * 255, 0), 255))

            pixels[x, ogl_y] = (r, g, b)
    return pixels

# Main loop.
running = True
clock = pygame.time.Clock()

# Initializing start time.
start_time = time.time()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    elapsed_time = time.time() - start_time
    iTime = float(f"{elapsed_time:.3f}")

    # Generate the UV shader pixel array.
    random_pixels = generate_uv_shader(canvas_width, canvas_height, iTime)

    # Create a pygame Surface from the numpy array.
    surface = pygame.surfarray.make_surface(random_pixels)

    # Blit the surface onto the canvas.
    canvas.blit(surface, (0, 0))

    # Update the display.
    pygame.display.flip()

    # Cap the frame rate to 120 FPS.
    clock.tick(120)

    pygame.display.set_caption(f"Splines UV Canvas | FPS: {clock.get_fps()}")

# Quit pygame
pygame.quit()
sys.exit()
