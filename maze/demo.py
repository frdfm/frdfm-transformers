"""Live token-by-token viewer. Never uses a search solver for model moves."""
import argparse
from pathlib import Path
import pygame
import torch
from .data import generate, advance, MOVES, BOS, END
from .model import load
from .train import checkpoint_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--frames', type=int, default=0, help='Exit after this many frames (smoke testing)')
    parser.add_argument('--autoplay', action='store_true')
    parser.add_argument('--size', type=int, default=11, help='Grid size; each size has its own trained checkpoint')
    parser.add_argument('--min-route', type=int, default=0, help='Only display mazes with at least this many solution moves')
    args = parser.parse_args()
    checkpoint = checkpoint_for(args.size)
    if not checkpoint.exists():
        parser.error(f'Train a checkpoint first: python -m maze.train --size {args.size} --steps 3000')
    torch.set_num_threads(2)
    model = load(checkpoint)
    pygame.init()
    screen = pygame.display.set_mode((1080, 760))
    pygame.display.set_caption('Maze Transformer | live token generation')
    font = pygame.font.SysFont('consolas', 19)
    big = pygame.font.SysFont('consolas', 32, bold=True)
    clock = pygame.time.Clock()
    seed = args.seed
    interval = 450
    paused = not args.autoplay
    running = True
    frames = 0
    last = pygame.time.get_ticks()

    def reset():
        nonlocal grid, position, goal, memory, tokens, trail, status, probabilities, done, seed, route_length
        for _ in range(10000):
            grid, position, goal, target = generate(seed, model.config['size'])
            route_length = len(target)-1
            if route_length >= args.min_route:
                break
            seed += 1
        else:
            raise ValueError('Could not find a maze meeting --min-route; reduce the threshold')
        with torch.inference_mode():
            memory = model.encode(torch.tensor([grid]))
        tokens = [BOS]
        trail = [position]
        status = 'Ready - SPACE to play, RIGHT to step'
        probabilities = [0.0]*5
        done = False

    def step():
        nonlocal position, status, probabilities, done
        if done:
            return
        with torch.inference_mode():
            probabilities = model.decode(torch.tensor([tokens]), memory)[0, -1].softmax(-1).tolist()
        action = max(range(5), key=probabilities.__getitem__)
        tokens.append(action)
        if action == END:
            status = 'SOLVED - model emitted END at goal' if position == goal else 'FAILED - END before reaching goal'
            done = True
        else:
            position, valid = advance(grid, position, action)
            if valid:
                trail.append(position)
                status = 'At goal - waiting for END' if position == goal else f'Generated {MOVES[action]}'
            else:
                status = f'FAILED - {MOVES[action]} hits a wall'
                done = True
        if not done and len(tokens)-1 >= model.config['size']**2:
            status, done = 'FAILED - move limit reached', True

    grid = position = goal = memory = tokens = trail = status = probabilities = done = route_length = None
    reset()

    def text(message, x, y, color=(206, 216, 231), heading=False):
        screen.blit((big if heading else font).render(message, True, color), (x, y))

    while running:
        single = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    last = pygame.time.get_ticks()
                elif event.key == pygame.K_RIGHT:
                    paused, single = True, True
                elif event.key == pygame.K_n:
                    seed += 1
                    reset()
                    last = pygame.time.get_ticks()
                elif event.key == pygame.K_r:
                    reset()
                    last = pygame.time.get_ticks()
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    interval = max(60, interval-100)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    interval = min(2000, interval+100)
        now = pygame.time.get_ticks()
        if single or (not paused and now-last >= interval):
            step()
            last = now
        screen.fill((14, 20, 32))
        text('MAZE / TRANSFORMER', 28, 22, heading=True)
        text('One maze in. One complete move sequence out.', 28, 65)
        cell = min(64, 460 // len(grid))
        ox, oy = 28, 117
        colors = [(38, 48, 65), (225, 232, 240), (70, 193, 141), (246, 181, 66)]
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                pygame.draw.rect(screen, colors[value], (ox+c*cell, oy+r*cell, cell-2, cell-2), border_radius=4)
        points = [(ox+c*cell+cell//2, oy+r*cell+cell//2) for r, c in trail]
        if len(points) > 1:
            pygame.draw.lines(screen, (58, 133, 220), False, points, 5)
        pygame.draw.circle(screen, (33, 95, 211), points[-1], cell//5)
        text(f'{args.size}x{args.size} | seed {seed}', 570, 114)
        text(f'Shortest route: {route_length} moves | emitted: {len(tokens)-1}', 570, 140)
        text('MODEL NEXT-TOKEN PROBABILITIES', 570, 173)
        for i, probability in enumerate(probabilities):
            y = 204+i*48
            text(MOVES[i], 570, y)
            pygame.draw.rect(screen, (38, 48, 65), (650, y, 290, 24), border_radius=4)
            pygame.draw.rect(screen, (70, 193, 141), (650, y, int(290*probability), 24), border_radius=4)
            text(f'{probability:5.1%}', 960, y)
        text('GREEDY DECODING / NO WALL MASK', 570, 477)
        text(f'{"PAUSED" if paused else "PLAYING"} | {interval} ms/token', 570, 511)
        text(status, 28, 592, (246, 181, 66))
        text('GENERATED TOKENS', 28, 629)
        recent = [MOVES[t] for t in tokens[1:]][-18:]
        for row in range(2):
            text(' > '.join(recent[row*9:(row+1)*9]), 28, 660+row*26)
        text('SPACE play/pause   RIGHT step   N new   R restart   +/- speed   ESC quit', 28, 724)
        pygame.display.flip()
        frames += 1
        if args.frames and frames >= args.frames:
            running = False
        clock.tick(60)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, str(args.screenshot))
    pygame.quit()


if __name__ == '__main__':
    main()
