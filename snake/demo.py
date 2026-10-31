"""Pygame viewer with the actual ten-move queue and projected head path."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random

from .game import ACTIONS, DELTAS, LEFT, RIGHT, STRAIGHT, Session
from .model import CHECKPOINT


def projected_path(game, moves):
    """Display-only simulation; never used to choose or correct model actions.

    Ignore future food placement, as the planner does after the known meal.
    """
    future = deepcopy(game)
    path = []
    for move in moves:
        destination, _ = future.destination(move)
        safe = future.safe(move)
        ate = future.step(move)
        path.append((destination, safe))
        if ate:
            future.food = None
        if not safe or future.won:
            break
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--player', default=str(CHECKPOINT), help='Checkpoint path, human, expert, or random')
    parser.add_argument('--seed', type=int, default=None, help='Reproduce a session; otherwise start randomly')
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--frames', type=int, default=0, help='Exit after N display frames')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.fps < 1 or args.frames < 0 or (args.headless and args.frames < 1):
        parser.error('Positive FPS required; headless mode requires --frames N')
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    import torch
    from .evaluate import action, controller

    torch.set_num_threads(2)
    try:
        player, config = controller(args.player)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(f'{exc}. Train first: python -m snake.prepare_data; python -m snake.train')
    session = Session(config, args.seed)
    game = session.game
    rng = random.Random(session.seed)
    pygame.init()
    cell = min(38, 600 // max(config.height, config.width))
    board_x, board_y = 24, 106
    panel = board_x + cell * config.width + 30
    screen = pygame.display.set_mode((panel + 382, max(690, config.height * cell + 195)))
    pygame.display.set_caption('Snake / 10-move Transformer')
    font = pygame.font.SysFont('consolas', 17)
    big = pygame.font.SysFont('consolas', 29, bold=True)
    small = pygame.font.SysFont('consolas', 13)
    clock = pygame.time.Clock()
    paused, numbers, running = False, False, True
    frames, fps, last_move, pending = 0, args.fps, STRAIGHT, STRAIGHT

    def text(value, x, y, color=(213, 226, 242), large=False, tiny=False):
        face = small if tiny else big if large else font
        screen.blit(face.render(value, True, color), (x, y))

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
                elif event.key == pygame.K_n:
                    paused, single = True, True
                elif event.key == pygame.K_m:
                    numbers = not numbers
                elif event.key == pygame.K_r:
                    game = session.restart()
                    rng.seed(session.seed)
                    pending = STRAIGHT
                    last_move = STRAIGHT
                    if player:
                        player.reset()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    fps = min(60, fps + 2)
                elif event.key == pygame.K_MINUS:
                    fps = max(1, fps - 2)
                elif args.player == 'human':
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        pending = LEFT
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        pending = RIGHT
        if running and (not paused or single) and game.alive and not game.won:
            last_move = pending if args.player == 'human' else action(args.player, player, game, rng)
            pending = STRAIGHT
            game.step(last_move)

        screen.fill((12, 18, 30))
        text('SNAKE / TRANSFORMER', 24, 20, large=True)
        text(f'Score {game.score:02d}   Length {len(game.body):02d}   Tick {game.ticks}', 24, 66)
        grid = game.matrix()
        colors = [(20, 31, 48), (61, 77, 99), (49, 157, 132), (250, 113, 113)] + [(99, 239, 186)] * 4
        for row in range(config.height):
            for col in range(config.width):
                value = int(grid[row, col])
                rect = (board_x + col*cell, board_y + row*cell, cell-2, cell-2)
                pygame.draw.rect(screen, colors[value], rect, border_radius=4)
                if numbers:
                    text(str(value), rect[0]+4, rect[1]+4, tiny=True)
        head = game.body[0]
        dr, dc = DELTAS[game.direction]
        center = (board_x + head[1]*cell + cell//2, board_y + head[0]*cell + cell//2)
        pygame.draw.line(screen, (12, 18, 30), center,
                         (center[0]+dc*cell//3, center[1]+dr*cell//3), 4)
        if player and game.alive and not game.won:
            for index, (p, safe) in enumerate(projected_path(game, player.queue[player.position:])):
                center = (board_x+p[1]*cell+cell//2, board_y+p[0]*cell+cell//2)
                pygame.draw.circle(screen, (255, 207, 94) if safe else (255, 94, 94), center, cell//3, 2)
                text(str(index+1), center[0]-5, center[1]-8, tiny=True)

        text('TEN-MOVE PLAN', panel, 107, (255, 207, 94), large=True)
        label = Path(args.player).stem if player else args.player.upper()
        text(f'Player: {label[:24]}', panel, 151)
        text(f'{config.history} x {config.height} x {config.width} matrix history', panel, 181)
        text('F forward  L left  R right', panel, 211)
        if player and player.queue:
            for index, move in enumerate(player.queue):
                x = panel + index % 5 * 68
                y = 252 + index // 5 * 72
                color = ((255, 207, 94) if index == player.position-1 else
                         (73, 89, 110) if index < player.position-1 else (99, 239, 186))
                pygame.draw.rect(screen, color, (x, y, 56, 57), width=2, border_radius=6)
                text(f'{index+1:02d}', x+6, y+4, color, tiny=True)
                text(('F', 'L', 'R')[move], x+18, y+20, color, large=True)
            text(f'Executed {player.position:02d}/10 | Plan {player.refills}', panel, 411)
            for index, probability in enumerate(player.probabilities):
                text(f'{ACTIONS[index]:7} {probability:5.1%}', panel, 449+index*26)
                pygame.draw.rect(screen, (99, 239, 186),
                                 (panel+190, 453+index*26, int(135*probability), 12))
        else:
            text(f'Action: {ACTIONS[last_move]}', panel, 260)
            text('No neural queue in this mode', panel, 295)
        status = 'BOARD CLEARED' if game.won else 'GAME OVER - R to restart' if not game.alive else 'PAUSED' if paused else 'PLAYING'
        text(f'{status} | {fps} ticks/s', panel, 550, (255, 207, 94))
        text('Wall/body paths masked in planning.' if player else 'R starts a new random board.', panel, 586, tiny=True)
        text(f'Seed {session.seed}', panel, 607, tiny=True)
        footer = board_y + cell*config.height + 20
        text('SPACE pause | N step | R restart | M matrix | +/- speed | ESC quit', 24, footer, tiny=True)
        text('Human: A/D turn | R starts a NEW random board.', 24, footer+24, tiny=True)
        pygame.display.flip()
        frames += 1
        if args.frames and frames >= args.frames:
            running = False
        if not args.headless:
            clock.tick(fps)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, str(args.screenshot))
    pygame.quit()


if __name__ == '__main__':
    main()
