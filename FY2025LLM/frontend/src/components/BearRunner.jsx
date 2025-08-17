// frontend/src/components/BearRunner.jsx
import React from "react";

export default function BearRunner({ scale = 4, speed = 120, color = "#747373" }) {
  const canvasRef = React.useRef(null);

  // 프레임 정의(가로/세로 자유. 단, 모든 프레임/모든 행 길이는 동일해야 함)
  const frames = React.useMemo(
    () => [
      [
        ".....###.....",
        "..##..#..##..",
        ".###########.",
        "..#########..",
        ".##.#####.##.",
        "###.#####.###",
        ".###########.",
        "..#########..",
        ".....#.#.....",
        "....#...#....",
        "...#.....#...",
        ".......#.....",
        "......#......",
      ],
      [
        ".....###.....",
        "..##..#..##..",
        ".###########.",
        "..#########..",
        ".###########.",
        "###.#####.###",
        ".###########.",
        "..#########..",
        ".....#.#.....",
        ".....#...#...",
        "......#..#...",
        ".......#.....",
        "......#......",
      ],
      [
        ".....###.....",
        "..##..#..##..",
        ".###########.",
        "..#########..",
        ".##.#####.##.",
        "###.#####.###",
        ".###########.",
        "..#########..",
        ".....#.#.....",
        "....#...#....",
        "...#.....#...",
        "......#......",
        ".......#.....",
      ],
      [
        ".....###.....",
        "..##..#..##..",
        ".###########.",
        "..#########..",
        ".###########.",
        "###.#####.###",
        ".###########.",
        "..#########..",
        ".....#.#.....",
        "...#...#.....",
        "....#..#.....",
        "......#......",
        ".......#.....",
      ],
    ],
    []
  );
  /*
  const frames = React.useMemo(
    () => [
      [
        ".....###.....",
        "..##..#..##..",
        ".###########.",
        "..#########..",
        ".##.#####.##.",
        "###.#####.###",
        ".###########.",
        "..#########..",
        ".....#.#.....",
        "....#...#....",
        "...#.....#...",
        ".......#.....",
        "......#......",
      ],
      [
        ".....###.....",
        "..##..#..##..",
        ".###########.",
        "..#########..",
        ".###########.",
        "###.#####.###",
        ".###########.",
        "..#########..",
        ".....#.#.....",
        ".....#...#...",
        "......#..#...",
        ".......#.....",
        "......#......",
      ],
      [
        ".....###.....",
        "..##..#..##..",
        ".###########.",
        "..#########..",
        ".##.#####.##.",
        "###.#####.###",
        ".###########.",
        "..#########..",
        ".....#.#.....",
        "....#...#....",
        "...#.....#...",
        "......#......",
        ".......#.....",
      ],
      [
        ".....###.....",
        "..##..#..##..",
        ".###########.",
        "..#########..",
        ".###########.",
        "###.#####.###",
        ".###########.",
        "..#########..",
        ".....#.#.....",
        "...#...#.....",
        "....#..#.....",
        "......#......",
        ".......#.....",
      ],
    ],
    []
  );


  */

  // 가로/세로 자동 대응 드로잉
  const drawFrame = React.useCallback(
    (ctx, frameIndex) => {
      const px = frames[frameIndex % frames.length];
      const h = px.length;         // 행 수(세로)
      const w = px[0].length;      // 열 수(가로) - 첫 행 길이 기준
      const p = scale;

      ctx.clearRect(0, 0, w * p, h * p);
      ctx.fillStyle = color;

      for (let y = 0; y < h; y++) {
        const row = px[y];
        for (let x = 0; x < w; x++) {
          if (row[x] === "#") ctx.fillRect(x * p, y * p, p, p);
        }
      }
    },
    [frames, scale, color]
  );

  // 프레임 유효성 체크(선택)
  React.useEffect(() => {
    const h0 = frames[0].length;
    const w0 = frames[0][0].length;
    const ok =
      frames.every((f) => f.length === h0) &&
      frames.every((f) => f.every((row) => row.length === w0));
    if (!ok) {
      // eslint-disable-next-line no-console
      console.warn("BearRunner: 모든 프레임의 가로/세로 크기가 동일해야 합니다.");
    }
  }, [frames]);

  // 캔버스 크기 설정 + 애니메이션 루프
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const h = frames[0].length;
    const w = frames[0][0].length;
    canvas.width = w * scale;
    canvas.height = h * scale;

    const ctx = canvas.getContext("2d", { alpha: true });
    ctx.imageSmoothingEnabled = false;

    let frame = 0;
    let rafId;
    let last = performance.now();

    const loop = (t) => {
      if (t - last >= speed) {
        drawFrame(ctx, frame++);
        last = t;
      }
      rafId = requestAnimationFrame(loop);
    };

    drawFrame(ctx, 0);
    rafId = requestAnimationFrame(loop);

    return () => cancelAnimationFrame(rafId);
  }, [frames, scale, speed, drawFrame]);


  return <canvas ref={canvasRef} className="bear-canvas" />;
}
