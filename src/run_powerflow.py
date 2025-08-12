# run_powerflow.py
import os
from power_flow import powerflow

if __name__ == "__main__":
    INPUT_DIR = r"C:\dev\ACPF\BFS\examples\ieee-13"
    OUTPUT_DIR = r"C:\dev\ACPF\BFS\results"

    # 출력 폴더 없으면 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 실행
    result = powerflow(
        input=INPUT_DIR,
        output=OUTPUT_DIR,
        tolerance=1e-6,
        max_iterations=30,
        display_summary=True,
        timestamp=False,          # 결과 파일명에 타임스탬프 포함하려면 True
        display_topology=False,   # 토폴로지 시각화 표시
        save_topology=False,      # 토폴로지 이미지를 파일로 저장
        graph_title="ieee-13 test",
        marker_size=1.5,
        verbose=1
    )

    # powerflow가 에러 문자열을 반환하는 경우 종료
    if isinstance(result, str) and result.lower().startswith("execution aborted"):
        raise SystemExit(result)

    print("Power flow finished. Results saved to:", OUTPUT_DIR)
