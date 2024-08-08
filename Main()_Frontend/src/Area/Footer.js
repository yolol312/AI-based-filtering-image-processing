import React, { useContext, useEffect, useState } from 'react';
import { ClipDataContext } from '../Data/ClipDataContext';
import './Footer.css';

const Server_IP = process.env.REACT_APP_Server_IP;

function Footer({ onClipSwapClick, clipInfo }) {
  const { clipInfo: clipInfoFromContext } = useContext(ClipDataContext);
  const [videos, setVideos] = useState([]);

  useEffect(() => {
    const clipInfoToUse = clipInfo || clipInfoFromContext;
    if (Array.isArray(clipInfoToUse) && clipInfoToUse.length > 0) {
      setVideos(clipInfoToUse.map((clip) => clip.clip_video || ''));
    } else {
      setVideos([]);
    }
  }, [clipInfo, clipInfoFromContext]);

  const handleVideoClick = (video) => {
    const videoUrl = `${Server_IP}/stream_clipvideo?clip_video=${encodeURIComponent(video)}`;
    console.log('Video URL:', videoUrl);
    onClipSwapClick(videoUrl);
  };

  if (videos.length === 0) {
    return (
      <div className="Footer">
        {/* 비디오가 없을 때의 표시 */}
      </div>
    );
  }

  return (
    <div className="Footer">
      <div className="video-list">
        {videos.map((video, index) => (
          <div key={index} className="video-item" onClick={() => handleVideoClick(video)}>
            {/* 비디오 아이템을 클릭할 수 있게 처리 */}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Footer;
