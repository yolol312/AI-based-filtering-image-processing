// Footer.js
import React, { useContext, useEffect, useState } from 'react';
import { DataContext } from '../Data/DataContext';
import './Footer.css';

// 환경 변수에서 서버 IP를 가져옵니다.
const Server_IP = process.env.REACT_APP_Server_IP;

function Footer({ onClipSwapClick }) {
  const { data } = useContext(DataContext);
  const [videos, setVideos] = useState([]);

  useEffect(() => {
    // data.clipInfo가 배열인지 확인하고 비디오 리스트를 업데이트합니다.
    if (Array.isArray(data.clipInfo)) {
      setVideos(data.clipInfo.map((clip) => clip.clip_video || ''));
    } else {
      setVideos([]); // data.clipInfo가 배열이 아니면 빈 배열로 설정
    }
  }, [data.clipInfo]);

  const getFileName = (path) => {
    // 경로를 '/'로 나누고 마지막 요소를 반환합니다.
    return path.split('/').pop();
  };

  const handleVideoClick = (video) => {
    const videoUrl = `${Server_IP}/stream_clipvideo?clip_video=${encodeURIComponent(video)}`;
    console.log('Video URL:', videoUrl);  // 콘솔에 링크 주소 출력

    // 비디오 URL을 그대로 onClipSwapClick에 전달
    onClipSwapClick(videoUrl);
  };

  if (videos.length === 0) {
    return (
      <div className="Footer">
      </div>
    );
  }

  return (
    <div className="Footer">
      <div className="video-list">
        {videos.map((video, index) => (
          <div
            key={index}
            className="video-item"
            onClick={() => handleVideoClick(video)}
          >
            {getFileName(video)} {/* 파일 이름만 표시 */}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Footer;
