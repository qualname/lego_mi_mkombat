import numpy
import pygame


DISPLAY_SCALE = 2


def init_display(initial_obs):
    height, width, *_ = initial_obs.shape
    output_dims = (width * DISPLAY_SCALE, height * DISPLAY_SCALE)
    return pygame.display.set_mode(output_dims)


def render(display, observation):
    height, width, *_ = observation.shape
    output_dims = (width * DISPLAY_SCALE, height * DISPLAY_SCALE)

    obs = numpy.transpose(observation, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    pygame.transform.scale(surface, output_dims, display)
    pygame.display.flip()

    pygame.event.pump()
