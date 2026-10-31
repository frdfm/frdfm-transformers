"""Scout real full games, edit close calls/endings, and render a highlight GIF."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import torch
from PIL import GifImagePlugin, Image, ImageChops, ImageDraw, ImageFont

from .game import DELTAS, Snake
from .demo import projected_path
from .model import CHECKPOINT, ModelPlayer, load

ROOT = Path(__file__).resolve().parent
BG = (12, 18, 30)
INK = (214, 227, 242)
GREEN = (99, 239, 186)
GOLD = (255, 207, 94)
RED = (250, 113, 113)
SIZE = (900, 690)


@dataclass
class Snapshot:
    tick: int
    body: tuple
    direction: int
    food: tuple | None
    score: int
    alive: bool
    queue: tuple
    position: int
    plan: int
    safe_moves: int
    adjacent_body: int
    wall_distance: int
    ate: bool
    turned: bool
    collision: tuple | None


def capture(game, player, ate=False, move=0, collision=None):
    r, c = game.body[0]
    nearby = sum(abs(r-br) + abs(c-bc) == 1 for br, bc in game.body[2:-1])
    return Snapshot(game.ticks, tuple(game.body), game.direction, game.food, game.score,
                    game.alive, tuple(player.queue), player.position, player.refills,
                    sum(game.safe(a) for a in range(3)) if game.alive else 0,
                    nearby, min(r-1, c-1, game.config.height-2-r, game.config.width-2-c),
                    ate, move != 0, collision)


def scout(model, config, games, max_ticks, seed):
    """Actual model-controlled play; every terminal state starts a fresh game."""
    rng = random.Random(seed)
    runs = []
    player = ModelPlayer(model)
    for number in range(games):
        run_seed = rng.getrandbits(63)
        game = Snake(run_seed, config)
        player.reset()
        player.refills = 0
        before = player.refills
        states, moves = [capture(game, player)], []
        for _ in range(max_ticks):
            move = player(game.observation() if player.needs_plan else None)
            destination, _ = game.destination(move)
            ate = game.step(move)
            moves.append(move)
            states.append(capture(game, player, ate, move, destination if not game.alive else None))
            if not game.alive or game.won:
                break
        run = dict(number=number+1, seed=run_seed, states=states, moves=moves,
                   score=game.score, ticks=game.ticks, died=not game.alive,
                   refills=player.refills-before)
        assert run['refills'] == (game.ticks+9)//10
        runs.append(run)
        print(f'Run {number+1:02d}/{games}: {game.ticks:4d} ticks, '
              f'{game.score:2d} food, {"crashed -> reset" if not game.alive else "time limit -> reset"}', flush=True)
    return runs


def replay(manifest, model, config):
    """Re-edit recorded moves without scouting again or changing any action."""
    runs = []
    for saved in manifest['runs']:
        game = Snake(saved['seed'], config)
        player = ModelPlayer(model)
        states = [capture(game, player)]
        for tick, move in enumerate(saved['moves']):
            if tick % 10 == 0:
                queue = saved['moves'][tick:tick+10]
                if len(queue) < 10:
                    generated, _ = model.generate(torch.tensor(game.observation()).unsqueeze(0))
                    full = generated[0].tolist()
                    assert full[:len(queue)] == queue
                    queue = full
                player.queue, player.position = queue, 0
                player.refills += 1
            player.position += 1
            destination, _ = game.destination(move)
            ate = game.step(move)
            states.append(capture(game, player, ate, move, destination if not game.alive else None))
        assert game.score == saved['score'] and game.ticks == saved['ticks']
        assert (not game.alive) == saved['died']
        runs.append(dict(saved, states=states))
    return runs


def overlaps(a, b, padding=16):
    return a['run'] == b['run'] and a['start'] <= b['end']+padding and b['start'] <= a['end']+padding


def select_clips(runs, count=6):
    """Keep context, avoid overlapping excerpts, and include distinct endings."""
    escapes, corners, endings = [], [], []
    for ri, run in enumerate(runs):
        states = run['states']
        if run['died']:
            endings.append(dict(run=ri, event=len(states)-1, start=max(0, len(states)-49),
                                end=len(states)-1, kind='ending', title='NOWHERE LEFT TO GO',
                                rank=run['score']+run['ticks']/1000))
        for index, state in enumerate(states):
            if index < 32 or index+40 >= len(states) or len(state.body) < 15:
                continue
            # A close call must resolve successfully; endings are selected separately.
            if not all(s.alive for s in states[index:index+25]):
                continue
            common = dict(run=ri, event=index, start=index-24, end=index+24)
            if state.safe_moves == 1 and (state.adjacent_body or state.wall_distance == 0):
                title = ('THREADING THE NEEDLE' if state.adjacent_body >= 2 else
                         'BETWEEN BODY AND WALL' if state.wall_distance == 0 else 'ONE WAY OUT')
                escapes.append(dict(**common, kind='escape', title=title,
                                    rank=state.adjacent_body*8 + len(state.body)/8 + int(state.turned)*2))
            if state.ate and state.wall_distance <= 1:
                corners.append(dict(**common, kind='corner', title='DINNER AT THE EDGE',
                                    rank=len(state.body)/8 + state.adjacent_body*3))
    selected = []
    def take(pool, number):
        added = 0
        for candidate in sorted(pool, key=lambda c: c['rank'], reverse=True):
            if added >= number or len(selected) >= count:
                break
            # Spread footage across games, then allow a second separated excerpt.
            if any(overlaps(candidate, old) for old in selected):
                continue
            if sum(old['run'] == candidate['run'] for old in selected) >= 2:
                continue
            if added == 0 or not any(old['run'] == candidate['run'] for old in selected):
                selected.append(candidate)
                added += 1
        return added
    ending_count = max(2, count//4)
    corner_count = max(1, count//4)
    take(escapes, max(1, count-ending_count-corner_count))
    take(corners, corner_count)
    take(endings, ending_count)
    for pool in (escapes, corners, endings):
        take(pool, count-len(selected))
    if not selected:
        raise ValueError('No highlights found; scout more/longer games')
    # Alternate kinds and distribute endings throughout, finishing with the strongest.
    escape_clips = sorted((c for c in selected if c['kind'] == 'escape'), key=lambda c: c['rank'])
    corner_clips = sorted((c for c in selected if c['kind'] == 'corner'), key=lambda c: c['rank'])
    live = []
    while escape_clips or corner_clips:
        for _ in range(2):
            if escape_clips:
                live.append(escape_clips.pop(0))
        if corner_clips:
            live.append(corner_clips.pop(0))
    dead = sorted((c for c in selected if c['kind'] == 'ending'), key=lambda c: c['rank'])
    ordered = []
    while live:
        ordered.extend(live[:3])
        del live[:3]
        if len(dead) > 1:
            ordered.append(dead.pop(0))
    return ordered + dead


class Renderer:
    def __init__(self, config):
        self.config = config
        self.cell = min(26, 520//max(config.height, config.width))
        self.x, self.y = 24, 116
        self.panel = self.x + self.cell*config.width + 28
        font = Path('C:/Windows/Fonts/consola.ttf')
        bold = Path('C:/Windows/Fonts/consolab.ttf')
        self.fonts = {size: ImageFont.truetype(str(font), size) for size in (13, 15, 18, 23)}
        self.bold = {size: ImageFont.truetype(str(bold), size) for size in (20, 28, 36)}

    def predictions(self, snapshot):
        """Use the app's exact display-only projection for pending queue tokens."""
        if not snapshot.alive or snapshot.position >= len(snapshot.queue):
            return []
        game = Snake(0, self.config)
        game.body, game.direction, game.food = list(snapshot.body), snapshot.direction, snapshot.food
        return projected_path(game, snapshot.queue[snapshot.position:])

    def card(self, title, subtitle, detail=''):
        frame = Image.new('RGB', SIZE, 'black')
        draw = ImageDraw.Draw(frame)
        draw.text((450, 263), 'SNAKE / TRANSFORMER', fill=GREEN, font=self.fonts[18], anchor='mm')
        draw.text((450, 321), title, fill=INK, font=self.bold[36], anchor='mm')
        draw.text((450, 378), subtitle, fill=GOLD, font=self.fonts[18], anchor='mm')
        draw.text((450, 420), detail, fill=(126, 145, 169), font=self.fonts[15], anchor='mm')
        return frame

    def scene(self, run, clip, index, clip_number, total):
        s = run['states'][index]
        frame = Image.new('RGB', SIZE, BG)
        d = ImageDraw.Draw(frame)
        d.text((24, 18), 'SNAKE / CLOSE CALLS', font=self.bold[28], fill=INK)
        d.text((24, 61), f'{clip_number:02d} / {total:02d}   {clip["title"]}', font=self.fonts[18], fill=GOLD)
        d.text((24, 89), f'RUN {run["number"]:02d}     SCORE {s.score:02d}     LENGTH {len(s.body):02d}', font=self.fonts[15], fill=INK)
        c, cell = self.config, self.cell
        occupied = set(s.body)
        for row in range(c.height):
            for col in range(c.width):
                p = (row, col)
                color = ((57, 73, 95) if row in (0, c.height-1) or col in (0, c.width-1) else
                         GREEN if p == s.body[0] else (49, 157, 132) if p in occupied else
                         RED if p == s.food else (20, 31, 48))
                rect = (self.x+col*cell, self.y+row*cell, self.x+(col+1)*cell-2, self.y+(row+1)*cell-2)
                d.rounded_rectangle(rect, radius=3, fill=color)
        row, col = s.body[0]
        center = self.x+col*cell+cell//2, self.y+row*cell+cell//2
        dr, dc = DELTAS[s.direction]
        d.line((center, (center[0]+dc*cell//3, center[1]+dr*cell//3)), fill=BG, width=3)
        for number, (p, safe) in enumerate(self.predictions(s), 1):
            cx, cy = self.x+p[1]*cell+cell//2, self.y+p[0]*cell+cell//2
            radius = cell//3
            color = GOLD if safe else RED
            d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=color, width=2)
            d.text((cx, cy), str(number), fill=INK, font=self.fonts[13], anchor='mm')
        if s.collision:
            r, col = s.collision
            cx, cy = self.x+col*cell+cell//2, self.y+r*cell+cell//2
            d.line((cx-7, cy-7, cx+7, cy+7), fill=RED, width=4)
            d.line((cx-7, cy+7, cx+7, cy-7), fill=RED, width=4)
        x = self.panel
        d.text((x, 120), '10 MOVES AHEAD', fill=GREEN, font=self.bold[20])
        d.text((x, 154), f'Plan {s.plan}  /  move {s.position:02d}', fill=INK, font=self.fonts[18])
        for i, move in enumerate(s.queue):
            qx, qy = x+(i%5)*57, 201+(i//5)*68
            color = GOLD if i == s.position-1 else (72, 88, 110) if i < s.position-1 else GREEN
            d.rounded_rectangle((qx, qy, qx+47, qy+55), radius=5, outline=color, width=2)
            d.text((qx+5, qy+4), f'{i+1:02d}', fill=color, font=self.fonts[13])
            d.text((qx+24, qy+34), ('F', 'L', 'R')[move], fill=color, font=self.bold[20], anchor='mm')
        d.text((x, 355), 'F forward   L/R turn', fill=(138, 157, 181), font=self.fonts[15])
        d.text((x, 379), 'Circles = next predicted cells', fill=GOLD, font=self.fonts[13])
        d.text((x, 404), 'ESCAPE ROUTES', fill=INK, font=self.fonts[15])
        d.text((x, 434), str(s.safe_moves) if s.alive else 'TRAPPED', fill=GOLD if s.safe_moves == 1 else RED if not s.alive else GREEN,
               font=self.bold[36])
        slow = abs(index-clip['event']) <= 7
        d.text((x, 503), '0.5x  /  SLOW MOTION' if slow else '1x  /  ACTUAL PLAY', fill=GOLD if slow else INK, font=self.fonts[15])
        d.text((x, 537), f'Game time {s.tick/8:5.1f}s', fill=INK, font=self.fonts[15])
        d.text((x, 570), 'ROUND OVER' if not s.alive else 'QUEUE EXECUTING', fill=RED if not s.alive else GREEN, font=self.fonts[18])
        d.text((24, 654), 'Real model play  /  clips from separate runs  /  fades mark every cut', fill=(138, 157, 181), font=self.fonts[13])
        progress = (index-clip['start'])/max(1, clip['end']-clip['start'])
        d.rectangle((x, 615, x+277, 619), fill=(43, 57, 77))
        d.rectangle((x, 615, x+int(277*progress), 619), fill=GOLD)
        return frame


class GifWriter:
    """Fixed-palette streaming GIF with delta rectangles and exact timing."""
    def __init__(self, path, palette, fps=16):
        self.path, self.palette, self.fps = Path(path), palette, fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix('.gif.tmp')
        self.stream = self.temporary.open('wb')
        self.previous = None
        self.frames = 0

    def append(self, rgb, repeat=1):
        frame = rgb.quantize(palette=self.palette, dither=Image.Dither.NONE)
        if self.previous is None:
            for block in GifImagePlugin.getheader(frame, info={'loop': 0})[0]:
                self.stream.write(block)
            box = (0, 0, *frame.size)
        else:
            box = ImageChops.difference(frame.convert('RGB'), self.previous.convert('RGB')).getbbox() or (0, 0, 1, 1)
        duration = (round((self.frames+repeat)*100/self.fps)-round(self.frames*100/self.fps))*10
        for block in GifImagePlugin.getdata(frame.crop(box), offset=box[:2], duration=duration, disposal=1):
            self.stream.write(block)
        self.previous = frame
        self.frames += repeat

    def close(self):
        self.stream.write(b';')
        self.stream.close()
        self.temporary.replace(self.path)


def fade(writer, frame, incoming, steps=12):
    black = Image.new('RGB', frame.size, 'black')
    for i in range(steps):
        amount = (i+1)/steps if incoming else 1-(i+1)/steps
        writer.append(Image.blend(black, frame, amount))


def render(runs, clips, config, output):
    renderer = Renderer(config)
    reference = renderer.scene(runs[clips[0]['run']], clips[0], clips[0]['event'], 1, len(clips))
    atlas = Image.new('RGB', (SIZE[0]*3, SIZE[1]*2))
    black = Image.new('RGB', SIZE, 'black')
    for i, factor in enumerate((1, .8, .6, .4, .2, 0)):
        atlas.paste(Image.blend(black, reference, factor), ((i%3)*SIZE[0], (i//3)*SIZE[1]))
    palette = atlas.quantize(colors=256, dither=Image.Dither.NONE)
    writer = GifWriter(output, palette)
    intro = renderer.card('CLOSE CALLS & LAST MOVES', f'{len(runs)} full runs  /  {sum(r["ticks"] for r in runs)/8/60:.1f} minutes scouted',
                          'Ten predicted moves. One commitment at a time.')
    fade(writer, intro, True)
    writer.append(intro, 28)
    fade(writer, intro, False)
    writer.append(black, 6)
    contact = []
    for number, clip in enumerate(clips, 1):
        run = runs[clip['run']]
        card = renderer.card(clip['title'], f'CLIP {number:02d} / {len(clips):02d}    -    RUN {run["number"]:02d}',
                             'Final moments' if clip['kind'] == 'ending' else 'A close call that survived')
        fade(writer, card, True, 6)
        writer.append(card, 10)
        fade(writer, card, False, 6)
        first = renderer.scene(run, clip, clip['start'], number, len(clips))
        fade(writer, first, True, 8)
        for index in range(clip['start'], clip['end']+1):
            frame = renderer.scene(run, clip, index, number, len(clips))
            writer.append(frame, 4 if abs(index-clip['event']) <= 7 else 2)
        writer.append(frame, 24 if clip['kind'] == 'ending' else 8)
        fade(writer, frame, False)
        writer.append(black, 6)
        contact.append(renderer.scene(run, clip, clip['event'], number, len(clips)))
        print(f'Edited {number}/{len(clips)}: {clip["title"]} / run {run["number"]} / '
              f'ticks {clip["start"]}-{clip["end"]}', flush=True)
    outro = renderer.card('ONE MORE RUN?', 'The next reset is a fresh start.', 'Every scene above came from actual model play.')
    fade(writer, outro, True)
    writer.append(outro, 24)
    fade(writer, outro, False)
    writer.append(black, 10)
    writer.close()
    sheet = Image.new('RGB', (900, 345*((len(contact)+1)//2)), BG)
    for i, frame in enumerate(contact):
        sheet.paste(frame.resize((450, 345), Image.Resampling.LANCZOS), ((i%2)*450, (i//2)*345))
    sheet.save(output.with_name(output.stem+'-scenes.jpg'), quality=92)
    return writer.frames/writer.fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--games', type=int, default=12)
    parser.add_argument('--max-ticks', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=30102026)
    parser.add_argument('--clips', type=int, default=6)
    parser.add_argument('--output', type=Path, default=ROOT/'highlights.gif')
    parser.add_argument('--reuse', type=Path, help='Re-edit an existing JSON move recording')
    parser.add_argument('--extra-games', type=int, default=0, help='Scout more games after replaying --reuse')
    args = parser.parse_args()
    if args.games < 3 or args.max_ticks < 100 or args.clips < 3:
        parser.error('Use at least three games/clips and 100 ticks per game')
    if args.extra_games < 0 or (args.extra_games and not args.reuse):
        parser.error('--extra-games requires --reuse and a nonnegative count')
    torch.set_num_threads(2)
    model, config = load(args.checkpoint)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.reuse:
        source = json.loads(args.reuse.read_text(encoding='utf-8'))
        if source['checkpoint_sha256'] != checkpoint_hash:
            parser.error('The recording was made with a different checkpoint')
        runs = replay(source, model, config)
        if args.extra_games:
            extra = scout(model, config, args.extra_games, args.max_ticks, args.seed)
            offset = len(runs)
            for i, run in enumerate(extra, offset+1):
                run['number'] = i
            if {run['seed'] for run in extra} & {run['seed'] for run in runs}:
                parser.error('Additional games repeat old seeds; choose a new --seed')
            runs.extend(extra)
    else:
        runs = scout(model, config, args.games, args.max_ticks, args.seed)
    clips = select_clips(runs, args.clips)
    if len(clips) < args.clips:
        parser.error(f'Found only {len(clips)} distinct clips; scout more games')
    duration = render(runs, clips, config, args.output)
    manifest = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                    seed=source['seed'] if args.reuse else args.seed,
                    additional_seed=args.seed if args.extra_games else None,
                    source_ticks=sum(r['ticks'] for r in runs), source_ticks_per_second=8,
                    games=len(runs), duration_seconds=duration,
                    numbered_predictions=True,
                    clips=clips, runs=[{k: v for k, v in run.items() if k != 'states'} for run in runs])
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {duration:.1f}s highlight reel: {args.output} '
          f'({args.output.stat().st_size/1024**2:.2f} MiB)', flush=True)


if __name__ == '__main__':
    main()
