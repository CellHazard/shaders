import pygame
import sys
import numpy as np
import numba
import math
from numba import jit

# Just how small you want the canvas to be.
divisor = 5

# Set canvas dimensions
canvas_width = int(1920 / divisor)
canvas_height = int(1080 / divisor)

# Initialize pygame
pygame.init()

# Create the canvas
canvas = pygame.display.set_mode((canvas_width, canvas_height))
pygame.display.set_caption("Splines UV Canvas Shader")


@jit(nopython=True)
def main_image(frag_coord, i_resolution, i_time):
    #Copyright splinesen. GitHub: https://github.com/splinesen
    
    min_intensity = 1.5
    max_intensity = 5.5
    combined_intensity = 50.0
    wave_zoom = 5.0
    wave_stretch = 1.5

    uv = (frag_coord * 1.0 - i_resolution) / i_resolution[1]
    uv0 = wave_zoom * uv

    final_col = np.zeros(3)

    uv0[1] += wave_stretch * math.sin(uv0[0] - (i_time * 0.75))

    line_intensity = min_intensity + (max_intensity * abs(np.mod(uv[0] + i_time, 2.0) - 1.0))
    glow_width = abs(line_intensity / (combined_intensity * uv0[1]))

    final_col += glow_width * np.array([1.0 + math.sin(i_time * 0.33),
                                        1.0 - math.sin(i_time * 0.33),
                                        1.0 - math.cos(i_time * 0.33)])

    return final_col


@jit(nopython=True)
def generate_uv_shader(canvas_width, canvas_height, iTime):
    # Generate pixel colors based on shader logic
    pixels = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    for y in range(canvas_height):
        for x in range(canvas_width):
            frag_coord = np.array([y, x], dtype=np.float32)
            i_resolution = np.array([canvas_width, canvas_height], dtype=np.float32)
            color = main_image(frag_coord, i_resolution, iTime / 10)
            pixels[y, x] = np.clip(color * 255, 0, 255).astype(np.uint8)
    return pixels

# Main loop
running = True
clock = pygame.time.Clock()

iTime = 1

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate the UV shader pixel array
    random_pixels = generate_uv_shader(canvas_width, canvas_height, iTime)

    # Because pygames clock gives me headache.
    iTime += 1

     # Create a pygame Surface from the numpy array
    surface = pygame.surfarray.make_surface(random_pixels)

    # Resize the surface to match the canvas dimensions
    surface = pygame.transform.scale(surface, (canvas_width, canvas_height))

    # Blit the surface onto the canvas
    canvas.blit(surface, (0, 0))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate to 60 FPS
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
