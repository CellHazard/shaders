import pygame
import sys
import numpy as np
import numba
import math
from numba import jit

'''
Copyright by splinesen.
GitHub: splinesen
'''

divisor = 1

# Set canvas dimensions
canvas_width = int(1920 / divisor)
canvas_height = int(1080 / divisor)

# Initialize pygame
pygame.init()

# Create the canvas
canvas = pygame.display.set_mode((canvas_width, canvas_height))
pygame.display.set_caption("UV Shader Canvas")

minIntensity = 1.5
maxIntensity = 9.5
combinedIntensity = 0.5
waveZoom = 4.0
waveStretch = 2.5

@jit(nopython=True)
def generate_uv_shader(canvas_width, canvas_height, iTime):
    pixels = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    for y in range(canvas_height):
        for x in range(canvas_width):

            # uv (Still needs fixing).
            u = ((y * 2.0) - canvas_height) / canvas_height
            v = ((x * 2.0) - canvas_width) / canvas_height

            # uv0
            u0 = waveZoom * u
            v0 = waveZoom * v

            # finalCol
            r = 0
            g = 0
            b = 0

            v0 += waveStretch * math.sin(u0 - (iTime * 2.25))

            lineIntensity = minIntensity + (maxIntensity * abs((u + iTime) % 2.0 - 1.0))
            
            glowWidth = abs(lineIntensity)

            if combinedIntensity * v0 != 0:
                glowWidth = abs(lineIntensity / (combinedIntensity * v0))

            r += glowWidth * (1.0 + math.sin(iTime * 0.33))
            g += glowWidth * (1.0 - math.sin(iTime * 0.33))
            b += glowWidth * (1.0 - math.cos(iTime * 0.33))

            pixels[y, x] = (r, g, b)
    return pixels

# Main loop
running = True
clock = pygame.time.Clock()

iTime = 0

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate the UV shader pixel array.
    random_pixels = generate_uv_shader(canvas_width, canvas_height, iTime)

    iTime += 0.01

     # Create a pygame Surface from the numpy array.
    surface = pygame.surfarray.make_surface(random_pixels)

    # Resize the surface to match the canvas dimensions.
    surface = pygame.transform.scale(surface, (canvas_width, canvas_height))

    # Blit the surface onto the canvas.
    canvas.blit(surface, (0, 0))

    # Update the display.
    pygame.display.flip()

    # Cap the frame rate to 120 FPS. Runs at 100~ FPS for me at 1080p resolution.
    clock.tick(120)

    pygame.display.set_caption(f"Splines UV Canvas | FPS: {clock.get_fps()}")

# Quit pygame
pygame.quit()
sys.exit()