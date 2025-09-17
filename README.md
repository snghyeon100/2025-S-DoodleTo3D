✏️Doodle to Magic
📢 2025년 여름학기 AIKU 활동으로 진행한 프로젝트입니다 (🥉 동상 수상!!)

📌 프로젝트 링크
Doodle-to-Magic

🏀 Vercel 배포 링크
hiyseo/doodle-to-magic

💫 모델
pokemon-scribble

소개
본 프로젝트는 개인의 낙서 그림을 기반으로 3D 캐릭터를 자동 생성하는 것을 목표로 합니다.

기존에는 낙서를 3D 실물로 제작하기 위해 많은 시간과 비용이 드는 수작업에 의존해야 했으며, 이 과정에서 낙서 고유의 창의성이 희석될 수 있었습니다. 이러한 문제를 해결하기 위해, 우리는 인공지능 기술을 활용하여 불완전한 낙서 형태를 해석해 완성도 높은 2D 캐릭터로 다듬고, 이를 기반으로 정확하고 매끄러운 3D 모델을 자동 생성하는 파이프라인을 구축하고자 합니다. 이 기술을 통해 누구나 자신의 상상력이 담긴 낙서를 손쉽게 디지털 창작물이나 실제 장난감으로 구현할 수 있게 될 것입니다.

구체적인 목표는 다음과 같습니다.

낙서 고유의 특성을 유지한 아마추어 스타일 2D 이미지 변환
낙서 고유의 특성을 유지한 포켓몬 스타일 2D 이미지 변환 후 3D 형태의 object로 변환
End-to-End 파이프라인 구성
방법론
Pipeline 스크린샷 2025-09-02 오후 11 20 14

본 프로젝트는 Amateur Dataset과 Pokemon Dataset을 통해 finetuning한 control_v11p_sd15_scribble 모델을 통해 사용자가 직접 그린 Doodle을 Childlike Style 2D Image와 Pokemon Style 2D Image로 변환합니다. 이후, 3D Model인 TripoSR을 통해 Pokemon Style 2D Image를 입체적인 3D mesh로 변환합니다.

Data pre-processing

스크린샷 2025-09-03 오전 12 18 44 스크린샷 2025-09-02 오후 10 11 10
Training(LoRA) 스크린샷 2025-09-02 오후 11 22 02

Finetuning Dataset
데이터셋	설명
AMATEUR dataset	낙서 그림, 그리고 이를 segmentation한 그림, caption이 pair로 있는 데이터셋
Poketmon dataset	poketmon 그림과 각 그림에 대한 caption이 달려있는 데이터

Prior Research
control_v11p_sd15_scribble: https://huggingface.co/lllyasviel/sd-controlnet-scribble
TripoSR: https://github.com/VAST-AI-Research/TripoSR
환경 설정
Conda Env
OS Env

Linux
2D inference

scribble-lora: envs/scribble-lora.yml
conda env create -f envs/scribble-lora.yml -n scribble-lora
3D inference

Fin: envs/Fin.yml
conda env create -f envs/Fin.yml -n Fin
local에서의 실행 방법
inputs 폴더 안에 input_{num}.png 형식으로 이미지 파일 준비 후 아래 script 실행
./run_test.sh {input number} "cute {object name} pokemon character

example - input_1.png
./run_test.sh 1 "cute tiger pokemon character"

예시 결과
Prompt
amateur prompt : a childlike crayon drawing, cute {input} character, no background
pokemon prompt : pokemon style, cute {input} pokemon character, no background

image image image image
From Pokemon 2D image to 3D (left and right)
image
Contribution
Amatuer Style 2D Image task 에서 Baseline에 비해 원본 낙서 고유의 형태 유지
Pokemon Style 2D Image task 에서 Baseline에 비해 Pokemon 고유의 특성 재현
Pokemon 3D task 에서 얼굴 앞 뒤가 똑같아 보이지 않는 문제 해결 및 obj file을 viewr에 업로드 했을 때 색감이 흐릿한 문제 해결
image
Limitation
Amateur Dataset prompt 및 segment 오류
Titan Gpu로 인한 제한적인 3D model
Pip library 호환성 문제로 인한 end-to-end pipleline 불가
세심한 Texture 구현 불가
정량적 평가 지표의 부재
팀원
신명경 : 실험 진행, 3D Modeling
김윤서 : 2D Modeling, Pipeline 및 배포
김태관 : 3D Modeling, Pipeline
백승현 : 3D Modeling, Research
