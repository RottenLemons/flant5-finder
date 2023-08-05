FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /program
COPY . /program
RUN pip install -r /program/requirements.txt
CMD ["bash", "InstructionDetector/RunScripts/Run1/Run1.sh"]