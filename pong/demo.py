"""Live Pong: model/model, model/expert, or human/model."""
import argparse
import os
from pathlib import Path
import random

import numpy as np
import pygame
import torch

from .evaluate import action, controller
from .game import ACTIONS, DOWN, Pong, STAY, UP
from .prediction import predicted_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', default='pong/checkpoints/seq10.pt')
    parser.add_argument('--right', default='pong/checkpoints/seq10.pt')
    parser.add_argument('--seed', type=int, default=10000000)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--frames', type=int, default=0, help='Exit after N display frames')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--gif', type=Path, help='Record a looping GIF of actual model play')
    parser.add_argument('--seconds', type=int, default=60, help='GIF match duration')
    parser.add_argument('--gif-fps', type=int, default=15)
    parser.add_argument('--headless', action='store_true', help='Render without opening a window')
    args = parser.parse_args()
    if args.fps < 1 or args.frames < 0:
        parser.error('FPS must be positive and frames nonnegative')
    if args.gif and (args.seconds < 1 or not 1 <= args.gif_fps <= min(50, args.fps)):
        parser.error('GIF seconds must be positive and GIF FPS must be between 1 and min(50, FPS)')
    if args.gif and 'human' in (args.left, args.right):
        parser.error('Offline GIF recording requires automated players')
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    torch.set_num_threads(2)
    try:
        lp, config = controller(args.left)
        rp, other = controller(args.right, config if lp else None)
        config = other if rp else config
    except FileNotFoundError as exc:
        parser.error(f'{exc}. Train first: python -m pong.prepare_data; python -m pong.train')
    game = Pong(args.seed, config)
    rng = random.Random(args.seed)
    pygame.init()
    cell = min(24, 576 // config.height, 768 // config.width)
    court_w, court_h = config.width * cell, config.height * cell
    screen = pygame.display.set_mode((court_w + 340, max(700, court_h + 150)))
    pygame.display.set_caption('Pong / Matrix Transformers')
    font = pygame.font.SysFont('consolas', 17)
    big = pygame.font.SysFont('consolas', 30, bold=True)
    small = pygame.font.SysFont('consolas', max(10, cell // 2))
    clock = pygame.time.Clock()
    paused, numbers, running = False, False, True
    frames, fps = 0, args.fps
    moves = [STAY, STAY]
    recorder = None
    if args.gif:
        from .gif import GifRecorder
        recorder = GifRecorder(args.gif, args.gif_fps)
        args.frames = args.fps * args.seconds
        print(f'Recording {args.seconds}s of real play to {args.gif}', flush=True)

    def text(message, x, y, color=(211, 224, 239), large=False):
        screen.blit((big if large else font).render(message, True, color), (x, y))

    while running:
        single = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_m:
                    numbers = not numbers
                elif event.key == pygame.K_r:
                    game = Pong(args.seed, config)
                    for player in (lp, rp):
                        if player:
                            player.reset()
                elif event.key == pygame.K_RIGHT:
                    paused, single = True, True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(120, fps + 10)
                elif event.key == pygame.K_MINUS:
                    fps = max(5, fps - 10)
        if not paused or single:
            keys = pygame.key.get_pressed()
            for side, (name, player) in enumerate(((args.left, lp), (args.right, rp))):
                if name == 'human':
                    up, down = (pygame.K_w, pygame.K_s) if side == 0 else (pygame.K_UP, pygame.K_DOWN)
                    moves[side] = UP if keys[up] else DOWN if keys[down] else STAY
                else:
                    moves[side] = action(name, player, game, side, rng)
            game.step(*moves)
        screen.fill((12, 18, 30))
        text('PONG / MATRIX TRANSFORMERS', 20, 18, large=True)
        text(f'{game.score[0]:02d}    :    {game.score[1]:02d}', court_w // 2 - 70, 58, large=True)
        grid = game.matrix()
        colors = [(18, 29, 46), (62, 78, 100), (83, 211, 175), (255, 207, 94)]
        for row in range(config.height):
            for col in range(config.width):
                value = int(grid[row, col])
                rect = (20 + col * cell, 100 + row * cell, cell - 1, cell - 1)
                pygame.draw.rect(screen, colors[value], rect)
                if numbers:
                    screen.blit(small.render(str(value), True, (140, 158, 180)), (rect[0] + 3, rect[1] + 3))
        ball = tuple(map(int, np.argwhere(grid == 3)[0]))
        predictions = predicted_cells(game.paddles,
                                      [player.queue if player else [] for player in (lp, rp)],
                                      [player.position if player else 0 for player in (lp, rp)], config, ball)
        for prediction in predictions:
            center = (20 + prediction['col']*cell + cell//2, 100 + prediction['row']*cell + cell//2)
            color = (83, 211, 175) if prediction['side'] == 0 else (117, 186, 255)
            pygame.draw.circle(screen, color, center, cell//3, 2)
            label = small.render(str(prediction['number']), True, (230, 239, 251))
            screen.blit(label, label.get_rect(center=center))
        x = court_w + 45
        text('NUMERIC INPUT', x, 100, (255, 207, 94))
        text(f'{config.history} x {config.height} x {config.width} matrices', x, 131)
        text('0 empty  1 wall', x, 164)
        text('2 paddle 3 ball', x, 190)
        text('Right player sees a', x, 226)
        text('mirrored matrix.', x, 251)
        text('Circles: queued positions', x, 276, (255, 207, 94))
        for side, (name, player) in enumerate(((args.left, lp), (args.right, rp))):
            y = 300 + side * 145
            label = Path(name).stem if player else name
            text(f'{"LEFT" if side == 0 else "RIGHT"}: {label[:20]}', x, y, (83, 211, 175))
            detail = (f'{ACTIONS[moves[side]]:4} {player.position:02d}/10  plan {player.refills}'
                      if player else f'Action: {ACTIONS[moves[side]]}')
            text(detail, x, y + 25)
            if player:
                for index, probability in enumerate(player.probabilities):
                    text(f'{ACTIONS[index]:4} {probability:5.1%}', x, y + 52 + index * 22)
                    pygame.draw.rect(screen, (83, 211, 175),
                                     (x + 140, y + 56 + index * 22, int(120 * probability), 10))
                for index, move in enumerate(player.queue):
                    color = ((255, 207, 94) if index == player.position - 1 else
                             (80, 95, 113) if index < player.position - 1 else (83, 211, 175))
                    text(('S', 'U', 'D')[move], x + index * 25, y + 120, color)
                    if index == player.position - 1:
                        pygame.draw.line(screen, color, (x + index * 25, y + 140),
                                         (x + index * 25 + 10, y + 140), 2)
        text(f'{"PAUSED" if paused else "PLAYING"} | {fps} ticks/s', x, 610)
        footer = ('SEQ2SEQ | predict 10 moves, execute the entire queue, then refill'
                  if recorder else 'SPACE pause | RIGHT step | M matrix | R reset | +/- speed | ESC quit')
        text(footer, 20, court_h + 120)
        pygame.display.flip()
        if recorder and frames * args.gif_fps // fps != (frames - 1) * args.gif_fps // fps:
            recorder.append(screen)
        frames += 1
        if recorder and frames % (fps * 10) == 0:
            print(f'{frames // fps}s / {args.seconds}s | score {game.score[0]}:{game.score[1]} '
                  f'| returns {sum(game.hits)}', flush=True)
        if args.frames and frames >= args.frames:
            running = False
        if not recorder:
            clock.tick(fps)
    if recorder:
        recorder.close()
        print(f'Saved {recorder.frames} GIF frames; {args.gif.stat().st_size / 1024**2:.2f} MiB', flush=True)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, str(args.screenshot))
    pygame.quit()


if __name__ == '__main__':
    main()
