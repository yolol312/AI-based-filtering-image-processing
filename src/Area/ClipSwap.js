import React, { useEffect, useRef } from 'react';
import './ClipSwap.css';

function ClipSwap({ videoUrl }) {
  const videoRef = useRef();

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load(); // URL 변경 시 비디오 리로드
    }
  }, [videoUrl]);

  if (!videoUrl) {
    return <p>비디오 URL이 없습니다.</p>;
  }

  return (
    <div className="ClipSwap">
      <h2>스트리밍 영상</h2>
      <video ref={videoRef} controls>
        <source src={videoUrl} type="video/mp4" />
        비디오를 재생할 수 없습니다.
      </video>
    </div>
  );
}

export default ClipSwap;
