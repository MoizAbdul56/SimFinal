import pygame, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# clock object that ensure that animation has the same
# on all machines, regardless of the actual machine speed.
clock = pygame.time.Clock()

def load_image():
    image = pygame.image.load("football.jpg")
    return image


class Ball(pygame.sprite.Sprite):
    def __init__(self, img):
        pygame.sprite.Sprite.__init__(self)
        '''
        self.image = pygame.Surface([width, height])
        self.rect = self.image.get_rect()
        self.image.fill(WHITE)
        cx = self.rect.centerx
        cy = self.rect.centery
        pygame.draw.circle(self.image, color, (width//2, height//2), cx, cy)
        self.rect = self.image.get_rect()
        '''
        self.image = pygame.image.load(img)
        self.rect = self.image.get_rect()

    def update(self):
        pass
    def rotate(self, angle):
        print(angle)
        self.image = pygame.transform.rotate(self.image, angle*(180*np.pi))
class Simulation:
    def __init__(self):
        self.paused = True # starting in paused mode
        self.gamma = -0.01
        self.gravity = 9.8
        self.mass = 0.4;
        self.dt =0.33
        self.r = np.array([0.22, 0.22])
        self.pos = [60,60]
        self.velocity=[0,0]
        self.inertiaT = np.identity(2)
        self.orientation = np.zeros(2)
        self.momentum = np.zeros(2)
        self.angularMomentum = np.zeros(2)
        self.cur_time = 0;
        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
        self.solver.set_f_params(self.gamma, self.gravity)

    def f(self, t, state, arg1, arg2):
        print(state)
        dx = state[2]
        dy = state[3]
        dvx = state[2]*arg1
        dvy = state[3]*arg1-arg2*self.mass

        return [dx, dy, dvx, dvy]

    def setup(self, angle_degrees):
        rad = np.radians(angle_degrees)
        self.collision(rad)
        print(self.pos, self.velocity)
        state = [self.pos[0], self.pos[1], self.velocity[0], self.velocity[1]]
        self.solver.set_initial_value(state, 0)
        self.trace_x = [self.pos[0]]
        self.trace_y = [self.pos[1]]

        self.angle = rad

    def step(self):
        self.cur_time += self.dt
        sol = self.solver.integrate(self.cur_time)
        print(sol)
        self.pos = sol[:2]
        self.velocity = sol[-2:]
        self.trace_x.append(sol[0])
        self.trace_y.append(sol[1])

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def collision(self, rad):
        kick_vel = np.array([100.0*np.cos(rad), 100.0*np.sin(rad)])
        foot = np.array([.0,.15])
        e = 0.5
        station = np.ones([2])
        station *= rad
        n = (station-foot)/np.abs(station-foot)
        j = ((1 + e)*kick_vel * n)/(1/self.mass + 1/1.4)
        vBall = 0 + (j*n)/self.mass
        omegaBall = 0 + np.cross((np.linalg.inv(self.inertiaT)*self.r), (j*n))
        self.velocity = vBall
        self.angularMomentum = self.inertiaT*omegaBall

    def rotation(self):
        return np.linalg.norm(self.angularMomentum/self.mass)

def sim_to_screen(win_height, x, y):
    '''flipping y, since we want our y to increase as we move up'''
    x += 10
    y += 10

    return x, win_height - y

def main():
    if (len(sys.argv) < 2):
        print("\nCorrect usage of this applicaton requires:")
        print("A command line argument for ball type")
        print("A command line argument for kick angle")
        print("Use command line argument ? for help\n")
        sys.exit()
    if (sys.argv[1] == "?"):
        print("\nUsage:")
        print("Ball types are: Football, Soccerball")
        print("Angle: between 0 and 90 in degrees\n")
        sys.exit()

    if (sys.argv[1] == "Football"):
        image = "football.png"
    elif (sys.argv[1] == "Soccerball"):
        image = "soccerball.png"
    else:
        image = "football.png"
    # initializing pygame
    pygame.init()

    # top left corner is (0,0)
    win_width = 1280
    win_height = 640
    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption('Kicking a Ball')

    # setting up a sprite group, which will be drawn on the
    # screen
    my_sprite = Ball(image)
    my_group = pygame.sprite.Group(my_sprite)

    # setting up simulation
    sim = Simulation()
    sim.setup(float(sys.argv[2]))

    print('--------------------------------')
    print('Usage:')
    print('Press (r) to start/resume simulation')
    print('Press (p) to pause simulation')
    print('Press (q) to force exit simulation')
    print('Press (space) to step forward simulation when paused')
    print('--------------------------------')

    while True:
        # 30 fps
        clock.tick(30)

        # update sprite x, y position using values
        # returned from the simulation
        my_sprite.rect.x, my_sprite.rect.y = sim_to_screen(win_height, sim.pos[0], sim.pos[1])

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            sim.pause()
            continue
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            sim.resume()
            continue
        else:
            pass

        # clear the background, and draw the sprites
        screen.fill(WHITE)
        my_group.update()
        my_group.draw(screen)
        pygame.display.flip()

        if sim.pos[1] <= 49.:
            pygame.quit()
            break

        # update simulation
        if not sim.paused:
            if sim.pos[1] <= 55.:
                pygame.quit()
                sys.exit(0)
            sim.step()
            angl = sim.rotation()
            my_sprite.rotate(angl)
        else:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                sim.step()

    plt.figure(1)
    plt.plot(sim.trace_x, sim.trace_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.title('2D projectile trajectory')
    plt.show()


if __name__ == '__main__':
    main()
