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

Runs at 60~ FPS at 720p on my high end PC.

NOTE:
I'm resorting to using a very janky way to manually define every 
vector attribute separately because lists caused massive FPS drops.
'''


# Set canvas dimensions (same as in ShaderToy).
canvas_width: int = 1280
canvas_height: int = 720

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


@jit(nopython=True)
def unit_vector(x, y, z):
    a = x * x
    b = y * y
    c = z * z

    d = a + b + c
    d = math.sqrt(d)

    return x / d, y / d, z / d

@jit(nopython=True)
def rotate_2d(angle):
    return math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle)

@jit(nopython=True)
def multiply_matrix(mA, mB, mC, mD, vecX, vecY):
    return (mA * vecX) + (mB * vecY), (mC * vecX) + (mD * vecY)

@jit(nopython=True)
def fract(vX, vY, vZ):
    return vX - math.floor(vX), vY - math.floor(vY), vZ - math.floor(vZ)

@jit(nopython=True)
def map(pX, pY, pZ):
    qX, qY, qZ = fract(pX, pY, pZ)

    qX = (qX * 2.0) - 1.0
    qY = (qY * 2.0) - 1.0
    qZ = (qZ * 2.0) - 1.0

    return math.sqrt(qX**2 + qY**2 + qZ**2) - 0.05

@jit(nopython=True)
def trace(oX, oY, oZ, rX, rY, rZ):
    t = 0.0

    for _ in range(0, 16):
        nX = oX + (rX * t)
        nY = oY + (rY * t)
        nZ = oZ + (rZ * t)
        
        d = map(nX, nY, nZ)
        t += d * .5
    
    return t


@jit(nopython=True, parallel=True)
def generate_uv_shader(canvas_width: int, canvas_height: int, iTime: float, mouseX, mouseY) -> np.ndarray:
    pixels: np.ndarray = np.zeros((canvas_width, canvas_height, 3), dtype=np.uint8)

    # Parallelize the outer loop.
    for y in numba.prange(canvas_height):
        # Map y-coordinate to OpenGL's coordinate system.
        ogl_y: int = (canvas_height - y) - 1

        for x in range(canvas_width):
            # uv.
            u: float = (((x + 0.5) * 2.0) - canvas_width) / canvas_height
            v: float = (((y + 0.5) * 2.0) - canvas_height) / canvas_height

            # r.
            rX, rY, rZ = unit_vector(u, v, 1.0)

            # o.
            oX = math.cos(iTime * 0.2) * 10.4
            oY = mouseY * 0.01
            oZ = math.sin(iTime * 0.2) * 3.6

            mA, mB, mC, mD = rotate_2d(-mouseX * 0.01)
            mat2_X, mat2_Y = multiply_matrix(mA, mB, mC, mD, rX, rZ)
            rX = mat2_X
            rZ = mat2_Y

            # t.
            t = trace(oX, oY, oZ, rX, rY, rZ)

            fog = 1.0 / (1.0 + t * t * 0.08)

            r = fog * (150 * 0.01)
            g = fog * (30 * 0.01)
            b = fog * (math.sin(iTime) + 1.5 * 2)

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

    mouseX, mouseY = pygame.mouse.get_pos()

    # Generate the UV shader pixel array.
    random_pixels: np.ndarray = generate_uv_shader(canvas_width, canvas_height, iTime, mouseX, mouseY)

    # Create a pygame Surface from the numpy array.
    surface: pygame.Surface = pygame.surfarray.make_surface(random_pixels)

    # Blit the surface onto the canvas.
    canvas.blit(surface, (0, 0))

    # Update the display.
    pygame.display.flip()

    # Cap the frame rate to 120 FPS.
    clock.tick(120)

    pygame.display.set_caption(f"Splines UV Canvas | FPS: {clock.get_fps()} | X: {mouseX} Y: {mouseY}")

# Quit pygame.
pygame.quit()
sys.exit()
