"""Numbered paddle projections with racket-first, earliest-token priority."""

def predicted_cells(paddles, queues, positions, config, ball=None):
    """Actual racket first; otherwise the earliest pending token owns a cell.

    Predict only paddle heights from real queued actions. No ball forecasting.
    """
    occupied = set()
    for side, col in enumerate((1, config.width-2)):
        top = round(paddles[side]-(config.paddle_height-1)/2)
        occupied.update((row, col) for row in range(top, top+config.paddle_height))
    if ball is not None:
        occupied.add(ball)
    predictions = []
    low, high = 1+(config.paddle_height-1)/2, config.height-2-(config.paddle_height-1)/2
    for side, col in enumerate((1, config.width-2)):
        y = paddles[side]
        for number, action in enumerate(queues[side][positions[side]:], 1):
            y = min(high, max(low, y+(0, -config.paddle_speed, config.paddle_speed)[action]))
            p = round(y), col
            if p not in occupied:
                predictions.append(dict(side=side, number=number, row=p[0], col=p[1]))
                occupied.add(p)
    return predictions

