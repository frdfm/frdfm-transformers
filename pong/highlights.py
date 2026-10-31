"""Cinematic highlights from actual ten-move Transformer Pong matches."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .game import Config, Pong
from .gif import GifRecorder
from .model import CHECKPOINT, ModelPlayer, load
from .prediction import predicted_cells

ROOT = Path(__file__).resolve().parent
SIZE = (1000, 720)
BG, INK = (12, 18, 30), (214, 227, 242)
GREEN, BLUE = (99, 239, 186), (117, 186, 255)
GOLD, RED = (255, 207, 94), (250, 113, 113)
FPS, TICKS_PER_SECOND = 15, 30


@dataclass
class Snapshot:
    tick: int
    x: float
    y: float
    paddles: tuple
    score: tuple
    queues: tuple
    positions: tuple
    plans: tuple
    scorer: int | None
    hit: int | None
    miss: int | None
    margin: float | None
    since_wall: int
    rally: int
    returns: int
    trail: tuple


def scout(model, config, games, ticks, seed):
    rng = random.Random(seed)
    matches = []
    for match_index in range(games):
        match_seed = rng.getrandbits(63)
        game = Pong(match_seed, config)
        players = [ModelPlayer(model), ModelPlayer(model)]
        states, moves, rallies = [], [], []
        rally, rally_start, returns, last_wall = 0, 0, 0, -1000
        trail = []
        for tick in range(ticks):
            # Both independent queues advance once per tick, and never flush at points.
            if players[0].needs_plan:
                assert players[1].needs_plan
                observations = np.stack([game.observation(side) for side in range(2)])
                queue, probabilities = model.generate(torch.tensor(observations))
                for side, player in enumerate(players):
                    player.queue = queue[side].tolist()
                    player.queue_probabilities = probabilities[side].tolist()
                    player.position = 0
                    player.refills += 1
            actions = [p() for p in players]
            old_x, old_y, old_vx, old_vy = game.x, game.y, game.vx, game.vy
            old_paddles, old_hits = game.paddles.copy(), game.hits.copy()
            # Recover the final scoring frame before Pong automatically serves again.
            final_x, final_y = old_x+old_vx, old_y+old_vy
            wall = final_y < 1 or final_y > config.height-2
            if final_y < 1:
                final_y = 2-final_y
            elif final_y > config.height-2:
                final_y = 2*(config.height-2)-final_y
            if wall:
                last_wall = tick
            scorer = game.step(*actions)
            hit = next((side for side in range(2) if game.hits[side] > old_hits[side]), None)
            low, high = 1+(config.paddle_height-1)/2, config.height-2-(config.paddle_height-1)/2
            paddles = tuple(min(high, max(low, old_paddles[s]+(0, -config.paddle_speed, config.paddle_speed)[actions[s]])) for s in range(2))
            miss, margin = None, None
            for side, plane in enumerate((2, config.width-3)):
                crossed = old_x > plane >= final_x if side == 0 else old_x < plane <= final_x
                if crossed and hit != side:
                    miss, margin = side, abs(final_y-paddles[side])-config.paddle_height/2
            if hit is not None:
                returns += 1
                margin = config.paddle_height/2-abs(game.y-paddles[hit])
            x, y = (final_x, final_y) if scorer is not None else (game.x, game.y)
            trail.append((x, y))
            trail = trail[-9:]
            states.append(Snapshot(tick+1, x, y, paddles, tuple(game.score),
                                   tuple(tuple(p.queue) for p in players), tuple(p.position for p in players),
                                   tuple(p.refills for p in players), scorer, hit, miss, margin,
                                   tick-last_wall, rally, returns, tuple(trail)))
            moves.append(actions)
            if scorer is not None:
                rallies.append(dict(start=rally_start, end=tick, returns=returns, scorer=scorer))
                rally += 1
                rally_start, returns, last_wall, trail = tick+1, 0, -1000, []
        if rally_start < ticks:
            rallies.append(dict(start=rally_start, end=ticks-1, returns=returns, scorer=None))
        assert all(p.refills == (ticks+9)//10 for p in players)
        matches.append(dict(number=match_index+1, seed=match_seed, states=states, moves=moves,
                            rallies=rallies, score=list(game.score), hits=list(game.hits),
                            refills=[p.refills for p in players]))
        print(f'Match {match_index+1:02d}/{games}: score {game.score[0]}:{game.score[1]}, '
              f'{sum(game.hits)} returns, {len(rallies)} rallies', flush=True)
    return matches


def select_clips(matches, count):
    pools = {kind: [] for kind in ('rally', 'save', 'bank', 'point')}
    for mi, match in enumerate(matches):
        for ri, rally in enumerate(match['rallies']):
            start, end = rally['start'], rally['end']
            if end-start < 75:
                continue
            states = match['states']
            def candidate(event, kind, title, rank):
                return dict(match=mi, rally=ri, event=event, start=max(start, event-120),
                            end=min(end, event+90), kind=kind, title=title, rank=rank)
            if rally['returns'] >= 4:
                item = candidate(end, 'rally', 'THE LONG EXCHANGE', rally['returns']*10+(end-start)/100)
                item['start'] = max(start, end-240)
                pools['rally'].append(item)
            for i in range(start+45, end+1):
                state = states[i]
                if state.hit is not None and state.margin is not None:
                    if state.margin < .7:
                        pools['save'].append(candidate(i, 'save', 'JUST ENOUGH RACKET', 5-state.margin+state.returns/10))
                    if state.since_wall <= 12:
                        pools['bank'].append(candidate(i, 'bank', 'WALL, THEN RACKET', 15-state.since_wall+state.returns/10))
                if state.miss is not None and state.margin is not None and 0 < state.margin < 1.4:
                    pools['point'].append(candidate(i, 'point', 'MISSED BY A FRACTION', 3-state.margin+state.returns/10))
            if rally['scorer'] is not None:
                pools['point'].append(candidate(end, 'point', 'THE RALLY BREAKS', rally['returns']/2))
    selected, used = [], set()
    ranked = {kind: sorted(pool, key=lambda c: c['rank'], reverse=True) for kind, pool in pools.items()}
    while len(selected) < count:
        added = False
        for kind in ('save', 'bank', 'rally', 'point'):
            while ranked[kind]:
                item = ranked[kind].pop(0)
                key = item['match'], item['rally']
                if key not in used:
                    selected.append(item)
                    used.add(key)
                    added = True
                    break
            if len(selected) >= count:
                break
        if not added:
            break
    return selected


def clip_frames(clip):
    # Normal playback samples two simulation ticks per 15-fps frame; slow motion
    # samples every tick. No scene crosses an automatic serve/reset.
    index = clip['start']
    while index <= clip['end']:
        slow = abs(index-clip['event']) <= 18
        yield index, slow
        index += 1 if slow else 2


def estimated_seconds(clips):
    frames = 58
    for clip in clips:
        frames += 30+sum(1 for _ in clip_frames(clip))+18+12+6
    return (frames+58)/FPS


class Renderer:
    def __init__(self, config):
        self.config = config
        self.cell, self.x, self.y, self.panel = 20, 24, 132, 696
        self.font = {n: ImageFont.truetype('C:/Windows/Fonts/consola.ttf', n) for n in (11, 13, 15, 18)}
        self.bold = {n: ImageFont.truetype('C:/Windows/Fonts/consolab.ttf', n) for n in (19, 26, 34)}

    def card(self, title, subtitle, detail=''):
        image = Image.new('RGB', SIZE, 'black')
        d = ImageDraw.Draw(image)
        for y, text, color, font in ((261, 'PONG / TRANSFORMERS', GREEN, self.font[18]),
                                      (323, title, INK, self.bold[34]),
                                      (382, subtitle, GOLD, self.font[18]),
                                      (424, detail, (138, 157, 181), self.font[15])):
            d.text((SIZE[0]//2, y), text, fill=color, font=font, anchor='mm')
        return image

    def scene(self, match, clip, index, number, total):
        s, c, cell = match['states'][index], self.config, self.cell
        image = Image.new('RGB', SIZE, BG)
        d = ImageDraw.Draw(image)
        d.text((24, 17), 'PONG / THE CLOSE CALLS', fill=INK, font=self.bold[26])
        d.text((24, 59), f'{number:02d}/{total:02d}  {clip["title"]}', fill=GOLD, font=self.font[18])
        d.text((24, 93), f'MATCH {match["number"]:02d}   RALLY {s.rally+1:02d}   RETURNS {s.returns:02d}', fill=INK, font=self.font[15])
        d.text((736, 22), f'{s.score[0]:02d} : {s.score[1]:02d}', fill=INK, font=self.bold[34])
        d.text((736, 72), 'LEFT             RIGHT', fill=(138, 157, 181), font=self.font[13])
        for row in range(c.height):
            for col in range(c.width):
                rect = (self.x+col*cell, self.y+row*cell, self.x+(col+1)*cell-1, self.y+(row+1)*cell-1)
                color = (57, 73, 95) if row in (0, c.height-1) else (19, 30, 47)
                d.rectangle(rect, fill=color)
        mid = self.x+c.width*cell//2
        for y in range(self.y+cell, self.y+(c.height-1)*cell, 24):
            d.line((mid, y, mid, y+11), fill=(54, 68, 89), width=2)
        # Historical ball trail only. Never portray a physical trajectory as a model prediction.
        for i, (x, y) in enumerate(s.trail[:-1]):
            cx, cy = self.x+(x+.5)*cell, self.y+(y+.5)*cell
            alpha = (i+1)/len(s.trail)
            color = tuple(int(a+(b-a)*alpha*.55) for a, b in zip(BG, GOLD))
            d.ellipse((cx-3, cy-3, cx+3, cy+3), fill=color)
        ball = (int(np.clip(round(s.y), 1, c.height-2)), int(np.clip(round(s.x), 0, c.width-1)))
        predictions = predicted_cells(s.paddles, s.queues, s.positions, c, ball)
        for prediction in predictions:
            cx, cy = self.x+(prediction['col']+.5)*cell, self.y+(prediction['row']+.5)*cell
            color = GREEN if prediction['side'] == 0 else BLUE
            d.ellipse((cx-8, cy-8, cx+8, cy+8), outline=color, width=2)
            d.text((cx, cy), str(prediction['number']), fill=INK, font=self.font[11], anchor='mm')
        # Rackets draw last and occupy whole game cells: no mixed labels or circles.
        for side, col in enumerate((1, c.width-2)):
            top = round(s.paddles[side]-(c.paddle_height-1)/2)
            color = GREEN if side == 0 else BLUE
            for row in range(top, top+c.paddle_height):
                rect = (self.x+col*cell, self.y+row*cell, self.x+(col+1)*cell-1, self.y+(row+1)*cell-1)
                d.rounded_rectangle(rect, radius=3, fill=color)
        if 0 <= s.x <= c.width-1:
            cx, cy = self.x+(s.x+.5)*cell, self.y+(s.y+.5)*cell
            d.rectangle((cx-7, cy-7, cx+7, cy+7), fill=GOLD)
        else:
            cx = self.x+4 if s.x < 0 else self.x+c.width*cell-4
            cy = self.y+(s.y+.5)*cell
            d.line((cx-6, cy-6, cx+6, cy+6), fill=RED, width=3)
            d.line((cx-6, cy+6, cx+6, cy-6), fill=RED, width=3)
        x = self.panel
        d.text((x, 135), 'TEN-MOVE COMMITMENT', fill=GOLD, font=self.bold[19])
        for side, color in enumerate((GREEN, BLUE)):
            y = 187+side*164
            d.text((x, y), ('LEFT' if side == 0 else 'RIGHT')+' / TRANSFORMER', fill=color, font=self.bold[19])
            d.text((x, y+31), f'Plan {s.plans[side]:03d}  Move {s.positions[side]:02d}/10', fill=INK, font=self.font[15])
            for i, move in enumerate(s.queues[side]):
                qx = x+i*27
                shade = GOLD if i == s.positions[side]-1 else (75, 90, 110) if i < s.positions[side]-1 else color
                d.rounded_rectangle((qx, y+65, qx+23, y+97), radius=3, outline=shade, width=1)
                d.text((qx+12, y+81), ('S', 'U', 'D')[move], fill=shade, font=self.font[15], anchor='mm')
            d.text((x, y+111), 'S stay   U up   D down', fill=(138, 157, 181), font=self.font[13])
        slow = abs(index-clip['event']) <= 18
        d.text((x, 529), '0.5x / SLOW MOTION' if slow else '1x / ACTUAL PLAY', fill=GOLD if slow else INK, font=self.font[15])
        status = ('LEFT SCORES' if s.scorer == 0 else 'RIGHT SCORES') if s.scorer is not None else 'RALLY IN PLAY'
        d.text((x, 566), status, fill=GOLD if s.scorer is not None else INK, font=self.bold[19])
        d.text((24, 640), 'Circles: queued paddle positions / racket first, earliest prediction next', fill=INK, font=self.font[13])
        d.text((24, 666), 'Ball trail: past motion / fades mark cuts / all actions from actual model play', fill=(138, 157, 181), font=self.font[13])
        progress = (index-clip['start'])/max(1, clip['end']-clip['start'])
        d.rectangle((x, 613, x+270, 617), fill=(43, 57, 77))
        d.rectangle((x, 613, x+int(270*progress), 617), fill=GOLD)
        return image


def fade(writer, image, incoming, steps=12):
    black = Image.new('RGB', SIZE, 'black')
    for i in range(steps):
        amount = (i+1)/steps if incoming else 1-(i+1)/steps
        writer.append_image(Image.blend(black, image, amount))


def render(matches, clips, config, output):
    renderer = Renderer(config)
    reference = renderer.scene(matches[clips[0]['match']], clips[0], clips[0]['event'], 1, len(clips))
    black = Image.new('RGB', SIZE, 'black')
    atlas = Image.new('RGB', (SIZE[0]*3, SIZE[1]*2))
    for i, alpha in enumerate((1, .8, .6, .4, .2, 0)):
        atlas.paste(Image.blend(black, reference, alpha), ((i%3)*SIZE[0], (i//3)*SIZE[1]))
    writer = GifRecorder(output, FPS, atlas.quantize(colors=256, dither=Image.Dither.NONE))
    intro = renderer.card('EVERY RETURN IS A COMMITMENT', f'{len(clips)} scenes from {len(matches)} matches',
                          'Two Transformers. Ten moves at a time.')
    fade(writer, intro, True)
    writer.append_image(intro, 28)
    fade(writer, intro, False)
    writer.append_image(black, 6)
    contact = []
    timelines = []
    for number, clip in enumerate(clips, 1):
        match = matches[clip['match']]
        card = renderer.card(clip['title'], f'SCENE {number:02d}/{len(clips):02d}  /  MATCH {match["number"]:02d}',
                             'A different rally. A fresh contest.')
        fade(writer, card, True, 6)
        writer.append_image(card, 10)
        fade(writer, card, False, 6)
        first = renderer.scene(match, clip, clip['start'], number, len(clips))
        fade(writer, first, True, 8)
        start_time = writer.frames/FPS
        event_time = None
        for index, slow in clip_frames(clip):
            image = renderer.scene(match, clip, index, number, len(clips))
            if event_time is None and index >= clip['event']:
                event_time = writer.frames/FPS
            writer.append_image(image)
        writer.append_image(image, 18)
        fade(writer, image, False)
        writer.append_image(black, 6)
        timelines.append(dict(start_seconds=start_time, event_seconds=event_time, end_seconds=writer.frames/FPS))
        contact.append(renderer.scene(match, clip, clip['event'], number, len(clips)))
        print(f'Edited {number:02d}/{len(clips)}: {clip["title"]}, match {match["number"]}, '
              f'rally {clip["rally"]+1}', flush=True)
    outro = renderer.card('ANOTHER SERVE. ANOTHER CHANCE.', 'Every rally above came from actual model play.',
                          'The next ten moves are already queued.')
    fade(writer, outro, True)
    writer.append_image(outro, 28)
    fade(writer, outro, False)
    writer.append_image(black, 6)
    writer.close()
    sheet = Image.new('RGB', (1000, 360*((len(contact)+1)//2)), BG)
    for i, image in enumerate(contact):
        sheet.paste(image.resize((500, 360), Image.Resampling.LANCZOS), ((i%2)*500, (i//2)*360))
    sheet.save(output.with_name(output.stem+'-scenes.jpg'), quality=92)
    return writer.frames/FPS, timelines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--games', type=int, default=12)
    parser.add_argument('--ticks', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=30102028)
    parser.add_argument('--clips', type=int, default=32)
    parser.add_argument('--min-seconds', type=float, default=310)
    parser.add_argument('--output', type=Path, default=ROOT/'highlights-cinematic.gif')
    args = parser.parse_args()
    if min(args.games, args.ticks, args.clips) < 1 or args.min_seconds < 0:
        parser.error('Use positive game, tick and clip counts, and nonnegative minimum duration')
    torch.set_num_threads(2)
    model, config = load(args.checkpoint)
    matches = scout(model, config, args.games, args.ticks, args.seed)
    count = args.clips
    while True:
        clips = select_clips(matches, count)
        if len(clips) < count:
            parser.error('Insufficient distinct rallies; scout more matches')
        if estimated_seconds(clips) >= args.min_seconds:
            break
        count += 2
    print(f'Selected {len(clips)} scenes, estimated {estimated_seconds(clips):.1f}s', flush=True)
    duration, timelines = render(matches, clips, config, args.output)
    assert abs(duration-estimated_seconds(clips)) < .01
    manifest = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    config=asdict(config), seed=args.seed, games=args.games, ticks_per_game=args.ticks,
                    source_ticks=args.games*args.ticks, source_ticks_per_second=TICKS_PER_SECOND,
                    duration_seconds=duration, fps=FPS, clips=clips, timelines=timelines,
                    prediction_priority='actual racket, then earliest pending prediction',
                    kinds=dict(Counter(c['kind'] for c in clips)),
                    matches=[{key: value for key, value in m.items() if key != 'states'} for m in matches])
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {duration:.1f}s: {args.output} ({args.output.stat().st_size/1024**2:.2f} MiB)', flush=True)


if __name__ == '__main__':
    main()
