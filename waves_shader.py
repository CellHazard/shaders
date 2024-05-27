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

Runs at 38~ FPS at 1080p on my high end PC.
FPS depends on the amount of splines added.

NOTE:
I'm resorting to using a very janky way to manually define every 
vector attribute separately because lists caused massive FPS drops.
'''


# Set canvas dimensions (same as in ShaderToy).
canvas_width: int = 1920
canvas_height: int = 1080

# Initialize pygame.
pygame.init()

# Create the canvas.
canvas = pygame.display.set_mode((canvas_width, canvas_height))
pygame.display.set_caption("UV Shader Canvas")

# Settings:
minIntensity: int = 3.5
maxIntensity: int = 9.5
combinedIntensity: int = 5000.0
waveZoom: int = 2.3
waveStretch: int = 0.23

@jit(nopython=True, parallel=True)
def generate_uv_shader(canvas_width: int, canvas_height: int, iTime: float) -> np.ndarray:
    pixels: np.ndarray = np.zeros((canvas_width, canvas_height, 3), dtype=np.uint8)

    sin_iTime: float = math.sin(iTime * 0.33)
    cos_iTime: float = math.cos(iTime * 0.33)
    
    # Parallelize the outer loop.
    for y in numba.prange(canvas_height):
        # Map y-coordinate to OpenGL's coordinate system.
        ogl_y: int = (canvas_height - y) - 1

        for x in range(canvas_width):
            # uv.
            u: float = (((x + 0.5) * 2.0) - canvas_width) / canvas_height
            v: float = (((y + 0.5) * 2.0) - canvas_height) / canvas_height

            # uv0.
            u0: float = waveZoom * u
            v0: float = waveZoom * v

            # finalCol.
            r: float = 0.0
            g: float = 0.0
            b: float = 0.0

            for i in range(2, 11):
                v0 += waveStretch * (math.sin(u0 - (iTime * math.sin(i)) - 0.36) + math.cos(u0 - (iTime * math.cos(i)) + 0.53))

                lineIntensity: float = minIntensity + (maxIntensity * abs((u + iTime) % 2.0 - 1.0))

                glowWidth: float = abs(lineIntensity / (combinedIntensity * v0))

                r += glowWidth * (i + 1)
                g += glowWidth * 3.0
                b += glowWidth * (i + 15.0)

            # Convert to RGB values in the range [0, 255].
            r = int(min(max(r * 255, 0), 255))
            g = int(min(max(g * 255, 0), 255))
            b = int(min(max(b * 255, 0), 255))



            pixels[x, ogl_y] = (r, g, b)
    return pixels

# Main loop.
running: bool = True
clock: pygame.time.Clock = pygame.time.Clock()

# Initializing start time.
start_time: float = time.time()

while running:
    # Handle events.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    elapsed_time: float = time.time() - start_time
    iTime: float = float(f"{elapsed_time:.3f}")

    # Generate the UV shader pixel array.
    random_pixels: np.ndarray = generate_uv_shader(canvas_width, canvas_height, iTime)

    # Create a pygame Surface from the numpy array.
    surface: pygame.Surface = pygame.surfarray.make_surface(random_pixels)

    # Blit the surface onto the canvas.
    canvas.blit(surface, (0, 0))

    # Update the display.
    pygame.display.flip()

    # Cap the frame rate to 120 FPS.
    clock.tick(120)

    pygame.display.set_caption(f"Splines UV Canvas | FPS: {clock.get_fps()}")

# Quit pygame.
pygame.quit()
sys.exit()
