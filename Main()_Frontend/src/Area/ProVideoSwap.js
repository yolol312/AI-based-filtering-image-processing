import React, { useEffect, useRef } from 'react';
import './ProVideoSwap.css';

function ProVideoSwap({ personData, videoUrl }) {
  const videoRef = useRef();

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load(); // URL 변경 시 비디오 리로드
    }
  }, [videoUrl]);

  if (!videoUrl) {
    return <p>비디오 URL이 없습니다.</p>;
  }

  if (!personData || personData.length === 0) {
    return <p>선택된 비디오에 대한 인물 정보가 없습니다.</p>;
  }

  return (
    <div className="ProVideoSwap">
      <video ref={videoRef} controls>
        <source src={videoUrl} type="video/mp4" />
        비디오를 재생할 수 없습니다.
      </video>
    </div>
  );
}

export default ProVideoSwap;
