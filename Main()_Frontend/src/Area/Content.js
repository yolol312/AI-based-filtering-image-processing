import React, { useEffect, useRef, useState, useContext } from "react";
import { DataContext } from "../Data/DataContext";
import { LogDataContext } from "../Data/LogDataContext";
import { ImageLogDataContext } from "../Data/ImageLogDataContext";
import { RealTimeDataContext } from "../Data/RealTimeDataContext";
import "./Content.css";

function Content() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const { data } = useContext(DataContext);
  const { realtimedata } = useContext(RealTimeDataContext);
  const { updateLogInfo, clearLogInfo } = useContext(LogDataContext);
  const { updateImageLogInfo, clearImageLogInfo } =
    useContext(ImageLogDataContext);
  const [stream, setStream] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [serverImage, setServerImage] = useState(null);
  const userId = data?.userId;

  const MAX_LOG_SIZE = 5 * 1024 * 1024; // 예: 5MB 임계값
  let logSize = 0;

  const Server_IP = process.env.REACT_APP_REALTIME_Server_IP;
  const RealTime_Server_IP = process.env.REACT_APP_REALTIME_Server_IP;

  useEffect(() => {
    async function getStream() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        setStream(stream);
      } catch (err) {
        console.error("Error accessing webcam: ", err);
      }
    }
    getStream();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (stream && videoRef.current && !isStreaming) {
      videoRef.current.srcObject = stream;
    }
  }, [stream, isStreaming]);

  const captureFrameAndSend = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append("image", blob);

        const data = {
          user_data: {
            user_id: userId,
          },
        };

        formData.append("json_data", JSON.stringify(data));

        fetch(`${Server_IP}/realtime_upload_image`, {
          method: "POST",
          body: formData,
        })
          .then((response) => response.json())
          .then((logdata) => {
            console.log(logdata);
            if ("frame" in logdata) {
              const serverImage = `data:image/jpeg;base64,${logdata.frame}`;
              setServerImage(serverImage);

              // person_data가 딕셔너리로 올 경우 그대로 사용
              const personDataDict =
                typeof logdata.person_data === "object" &&
                !Array.isArray(logdata.person_data)
                  ? logdata.person_data
                  : {};

              const imageDataSize = logdata.frame.length * 2;
              logSize += imageDataSize;
              const personImageBase64Dict = {};

              // logdata.person_image가 딕셔너리 형태인 경우
              if (
                typeof logdata.person_image === "object" &&
                !Array.isArray(logdata.person_image)
              ) {
                for (const [key, value] of Object.entries(
                  logdata.person_image
                )) {
                  // value가 Uint8Array와 같은 바이너리 데이터인 경우 base64로 변환
                  if (value instanceof Uint8Array) {
                    personImageBase64Dict[key] = btoa(
                      new Uint8Array(value).reduce(
                        (data, byte) => data + String.fromCharCode(byte),
                        ""
                      )
                    );
                  } else if (typeof value === "string") {
                    // 만약 value가 이미 base64 문자열이라면 그대로 사용
                    personImageBase64Dict[key] = value;
                  } else {
                    console.warn(`Unexpected data type for key ${key}:`, value);
                  }
                }
                clearImageLogInfo();
                updateImageLogInfo({
                  timestamp: new Date().toISOString(),
                  image: logdata.frame,
                  message: logdata.message,
                  person_data: personDataDict, // 딕셔너리 형태 그대로 전달
                  person_image: personImageBase64Dict, // 딕셔너리 형태 그대로 전달
                });
              } else {
                console.log("응답에 프레임키가 없습니다.");
                updateLogInfo({
                  timestamp: new Date().toISOString(),
                  message: "Frame key is not present in the response.",
                  person_data: logdata.person_data,
                });
              }
            }
          })
          .catch((err) => {
            console.error(err);
          });
      }, "image/jpeg");
    }
  };

  const captureFrameAndEnd = () => {
    if (videoRef.current && canvasRef.current) {
      const data = {
        user_data: {
          user_id: userId,
        },
      };

      fetch(`${RealTime_Server_IP}/realtime_upload_image_end`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })
        .then((response) => response.json())
        .then((data) => {
          //console.log("Streaming ended successfully:", data);
        })
        .catch((err) => {
          console.error("Error ending stream: ", err);
        });
    }
  };

  useEffect(() => {
    let interval;
    if (isStreaming) {
      interval = setInterval(() => captureFrameAndSend(), 1000 / 30);
    }

    return () => clearInterval(interval);
  }, [isStreaming]);

  const startStreaming = () => {
    setIsStreaming(true);
    if (videoRef.current) {
      videoRef.current.style.display = "block";
    }
    if (canvasRef.current) {
      canvasRef.current.style.display = "block";
    }
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    setServerImage(null);

    if (videoRef.current) {
      videoRef.current.style.display = "block";
    }
    if (canvasRef.current) {
      canvasRef.current.style.display = "none";
    }
    captureFrameAndEnd();

    // 실시간 종료 시 로그 데이터 초기화
    clearLogInfo(); // 로그 데이터를 초기화합니다.
    clearImageLogInfo();
  };

  useEffect(() => {
    if (serverImage && canvasRef.current && videoRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const img = new Image();
      img.src = serverImage;

      img.onload = () => {
        if (videoRef.current) {
          canvas.width = videoRef.current.videoWidth;
          canvas.height = videoRef.current.videoHeight;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }
      };
    }
  }, [serverImage]);

  return (
    <div className="content">
      <div className="button-container">
        <button className="control-button" onClick={startStreaming}>
          실시간 시작
        </button>
        <button className="control-button" onClick={stopStreaming}>
          실시간 종료
        </button>
      </div>
      <div className="main-container">
        <div className="video-container">
          <h3>원본 영상</h3>
          <video ref={videoRef} autoPlay className="video-player" />
        </div>
        <div className="video-container">
          <h3>실시간 분석 영상</h3>
          <canvas ref={canvasRef} className="video-canvas" />
        </div>
      </div>
    </div>
  );
}

export default Content;
