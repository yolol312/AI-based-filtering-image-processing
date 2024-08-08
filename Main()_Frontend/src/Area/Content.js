// Content.js

import React, { useEffect, useRef } from "react";
import "./Content.css";

function Content() {
  const liveVideoRef1 = useRef(null);
  const liveVideoRef2 = useRef(null);
  const liveVideoRef3 = useRef(null);
  const liveVideoRef4 = useRef(null);

  useEffect(() => {
    const setupWebcam = (videoElement) => {
      if (videoElement) {
        navigator.mediaDevices
          .getUserMedia({ video: { deviceId: videoElement.dataset.deviceId } })
          .then((stream) => {
            videoElement.srcObject = stream;
          })
          .catch((error) => {
            console.error("Error accessing media devices.", error);
          });
      }
    };

    // 각 비디오 플레이어에 웹캠 스트림 연결
    setupWebcam(liveVideoRef1.current);
    /*setupWebcam(liveVideoRef2.current);
    setupWebcam(liveVideoRef3.current);
    setupWebcam(liveVideoRef4.current);*/

    return () => {
      // 컴포넌트 언마운트 시 스트림 정지
      [liveVideoRef1, liveVideoRef2, liveVideoRef3, liveVideoRef4].forEach(
        (ref) => {
          if (ref.current && ref.current.srcObject) {
            ref.current.srcObject.getTracks().forEach((track) => track.stop());
          }
        }
      );
    };
  }, []);

  return (
    <div className="content">
      <div className="video-container">
        <video
          ref={liveVideoRef1}
          className="video-player"
          controls
          muted
          playsInline
          autoPlay
        />
      </div>
      <div className="video-container">
        <video
          ref={liveVideoRef2}
          className="video-player"
          controls
          muted
          playsInline
          autoPlay
        />
      </div>
      <div className="video-container">
        <video
          ref={liveVideoRef3}
          className="video-player"
          controls
          muted
          playsInline
          autoPlay
        />
      </div>
      <div className="video-container">
        <video
          ref={liveVideoRef4}
          className="video-player"
          controls
          muted
          playsInline
          autoPlay
        />
      </div>
    </div>
  );
}

export default Content;