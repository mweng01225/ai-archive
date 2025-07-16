# test_nodes.py

import numpy as np
from connect5 import Node as NodeV1
from connect6 import Node as NodeV2

def expansion_positions(root, use_player_encoding=False):
    """
    Repeatedly call add_child() on `root` until it's fully expanded.
    Return a list of (row, col) coords where each new piece was dropped.
    If use_player_encoding=True, we expect root.board uses ±1; otherwise 1/2.
    """
    positions = []
    while not root.expanded:
        before = root.board.copy()
        root.add_child()
        # the newly‐added child is always at the end
        child = root.children[-1]
        diff = child.board - before

        # find the one entry that changed
        # for v1: diff is 1 or 2; for v2: diff is ±1
        nz = np.argwhere(diff != 0)
        assert nz.shape[0] == 1, "Should only ever add one piece"
        positions.append(tuple(nz[0]))
    return positions

def main():
    # 1) Test V1
    empty = np.zeros((6,7), dtype=int)
    root1 = NodeV1(None, empty.copy(), turn=1)
    pos1 = expansion_positions(root1, use_player_encoding=False)
    print("V1 expansion sequence:", pos1)

    # 2) Test V2
    # if your second Node expects `player` in {–1,+1}, 
    # pass in –1 so that internally self.player becomes +1 on the root
    root2 = NodeV2(None, empty.copy(), player=-1)
    pos2 = expansion_positions(root2, use_player_encoding=True)
    print("V2 expansion sequence:", pos2)

    # 3) Compare
    if pos1 == pos2:
        print(" Both implementations expand in exactly the same order.")
    else:
        print(" Mismatch!")
        print(" V1:", pos1)
        print(" V2:", pos2)

if __name__ == "__main__":
    main()
