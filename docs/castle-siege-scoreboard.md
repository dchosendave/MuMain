# Castle Siege scoreboard

On servers which run Castle Siege as a point battle (OpenMU's "Castle Siege: point mode" plugin), a
scoreboard shows on the siege map in Valley of Loren while the battle runs, above the siege mini map.

It lists up to eight guilds, best first, with their points. A star marks the guild holding the
castle, and the hero's guild is highlighted. Below the list:

- **Your guild**: the hero's guild (its alliance's main guild) and its place.
- **Next tick**: the time until the castle holder scores again for holding the castle.

The title row counts the battle's remaining time down. When the battle ends, the board shows the
final scores for 30 seconds, then closes. It also closes when the hero leaves the siege map; the
server sends the current scores again on the way back in.

## Scoring rules

The server decides the points; the client only shows them. With the default settings:

| Event | Points |
|---|---|
| Sealing the crown (the guild takes the castle) | 10 |
| Killing an enemy who held a switch for at least 3 seconds | 2 |
| Every full 5 minutes the guild holds the castle (the clock restarts with each seal) | 3 |

The guild with the most points wins and takes the castle; on a tie, the guild holding the castle at
the end wins. Server administrators change the values on the player website or the admin panel.

Unlike the original rules, where whoever holds the castle when the time runs out wins, a seal in the
last second is worth one seal, not the whole battle.

## Protocol

The server sends the board with the custom packet `C2 FB 01` whenever the score changes, every 30
seconds to keep the timers right, and to a player entering the siege map. The client counts the
timers down itself between packets. Servers without the point mode never send it, so the board
never shows there.
