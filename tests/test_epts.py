"""EPTS reader on a tiny synthetic file: column mapping, substitutions, NaN handling."""

import os
import tempfile
import unittest

import numpy as np

from mtah.epts import read_epts

META = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
  <Teams><Team id="T1"/><Team id="T2"/></Teams>
  <Players>
    <Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/><Player id="P3" teamId="T1"/>
  </Players>
  <PlayerChannels>
    <PlayerChannel id="p1x" playerId="P1" channelId="x"/>
    <PlayerChannel id="p1y" playerId="P1" channelId="y"/>
    <PlayerChannel id="p1s" playerId="P1" channelId="speed"/>
    <PlayerChannel id="p2x" playerId="P2" channelId="x"/>
    <PlayerChannel id="p2y" playerId="P2" channelId="y"/>
    <PlayerChannel id="p2s" playerId="P2" channelId="speed"/>
    <PlayerChannel id="p3x" playerId="P3" channelId="x"/>
    <PlayerChannel id="p3y" playerId="P3" channelId="y"/>
    <PlayerChannel id="p3s" playerId="P3" channelId="speed"/>
  </PlayerChannels>
</Metadata>
<DataFormatSpecifications>
  <DataFormatSpecification startFrame="1" endFrame="2">
    <StringRegister name="frameCount"/>
    <SplitRegister separator=";">
      <SplitRegister separator=","><PlayerChannelRef playerChannelId="p1x"/>
        <PlayerChannelRef playerChannelId="p1y"/><PlayerChannelRef playerChannelId="p1s"/>
      </SplitRegister>
      <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2x"/>
        <PlayerChannelRef playerChannelId="p2y"/><PlayerChannelRef playerChannelId="p2s"/>
      </SplitRegister>
    </SplitRegister>
    <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>
      <BallChannelRef channelId="z"/></SplitRegister>
  </DataFormatSpecification>
  <DataFormatSpecification startFrame="3" endFrame="3">
    <StringRegister name="frameCount"/>
    <SplitRegister separator=";">
      <SplitRegister separator=","><PlayerChannelRef playerChannelId="p3x"/>
        <PlayerChannelRef playerChannelId="p3y"/><PlayerChannelRef playerChannelId="p3s"/>
      </SplitRegister>
      <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2x"/>
        <PlayerChannelRef playerChannelId="p2y"/><PlayerChannelRef playerChannelId="p2s"/>
      </SplitRegister>
    </SplitRegister>
    <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>
      <BallChannelRef channelId="z"/></SplitRegister>
  </DataFormatSpecification>
</DataFormatSpecifications></main>
"""

RAW = """1:0.1,0.2,1.0;0.3,0.4,2.0:0.5,0.5,0.0
2:NaN,NaN,NaN;0.31,0.41,2.0:NaN,NaN,NaN
3:0.9,0.8,3.0;0.32,0.42,1.0:0.6,0.4,0.1
"""


class EptsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.meta = os.path.join(self.dir.name, "m.xml")
        self.raw = os.path.join(self.dir.name, "r.txt")
        with open(self.meta, "w") as fh:
            fh.write(META)
        with open(self.raw, "w") as fh:
            fh.write(RAW)

    def tearDown(self):
        self.dir.cleanup()

    def test_mapping_and_substitution(self):
        d = read_epts(self.meta, self.raw)
        np.testing.assert_array_equal(d["frame_id"], [1, 2, 3])
        np.testing.assert_array_equal(d["team"], [0, 1, 0])
        self.assertAlmostEqual(float(d["px"][0, 0]), 0.1, places=5)
        self.assertAlmostEqual(float(d["py"][1, 1]), 0.41, places=5)
        # P3 only appears in the second spec, in the first column group
        self.assertTrue(np.isnan(d["px"][0, 2]))
        self.assertAlmostEqual(float(d["px"][2, 2]), 0.9, places=5)
        self.assertTrue(np.isnan(d["px"][2, 0]))

    def test_missing_values(self):
        d = read_epts(self.meta, self.raw)
        self.assertTrue(np.isnan(d["px"][1, 0]))
        self.assertTrue(np.isnan(d["bx"][1]))
        self.assertAlmostEqual(float(d["bx"][2]), 0.6, places=5)

    def test_column_mismatch_is_structural_error(self):
        with open(self.raw, "w") as fh:
            fh.write("1:0.1,0.2;0.3,0.4:0.5,0.5\n")
        with self.assertRaises(ValueError) as ctx:
            read_epts(self.meta, self.raw)
        self.assertIn("columns", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
