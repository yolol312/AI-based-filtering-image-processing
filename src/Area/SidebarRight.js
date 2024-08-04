// SidebarRight.js
import React, { useContext, useState } from 'react';
import './SidebarRight.css';
import { DataContext } from '../Data/DataContext';
import { ClipDataContext } from '../Data/ClipDataContext'; // ClipDataContext 추가

// 환경 변수에서 서버 IP를 가져옵니다.
const Server_IP = process.env.REACT_APP_Server_IP;

function SidebarRight({ onPersonSelect }) {
  const { data, setData } = useContext(DataContext);
  const { updateClipInfo } = useContext(ClipDataContext); // ClipDataContext에서 updateClipInfo 가져오기
  const userId = data.userId;
  const [selectedVideo, setSelectedVideo] = useState(null);

  const groupedData = data.provideopersoninfo.reduce((acc, person) => {
    const videoName = person.pro_video_name;
    if (!acc[videoName]) {
      acc[videoName] = [];
    }
    acc[videoName].push(person);
    return acc;
  }, {});

  const handleVideoNameClick = async (videoName, persons) => {
    const payload = {
      user_id: userId,
      pro_video_name: videoName
    };
  
    const queryParams = new URLSearchParams(payload).toString();
    const videoUrl = `${Server_IP}/stream_video?${queryParams}`;
  
    console.log('Selected Video URL:', videoUrl);
    console.log('Persons Data:', persons);
  
    try {
      // 비디오 URL을 설정하여 비디오를 재생
      setSelectedVideo({ videoName, persons, videoUrl });
      onPersonSelect(persons, videoUrl);

      // ClipDataContext에 클립 정보 업데이트
      updateClipInfo(persons.map(person => ({
        clip_video: videoUrl, // 클립 비디오 URL 저장
        person_id: person.person_id,
        person_details: person // 추가 정보 저장
      })));
    } catch (error) {
      console.error('Error sending data to server:', error);
    }
  };
  
  const handlePersonClick = async (person) => {
    try {
      const response = await fetch(`${Server_IP}/get_Person_to_clip?user_id=${userId}&person_id=${person.person_id}`);
      if (response.ok) {
        const data = await response.json();
        setData({ clipInfo: data.clip_info });
        console.log('Person data saved:', data);
      } else {
        console.error('Failed to fetch person data');
      }
    } catch (error) {
      console.error('Error fetching person data:', error);
    }
  };

  return (
    <div className="SidebarRight">
      <h2 className="SidebarRighttitle">비디오 목록</h2>
      {Object.entries(groupedData).map(([videoName, persons], index) => (
        <VideoCard
          key={index}
          videoName={videoName}
          persons={persons}
          onVideoNameClick={() => handleVideoNameClick(videoName, persons)}
          onPersonClick={handlePersonClick}
          isSelected={selectedVideo && selectedVideo.videoName === videoName}
        />
      ))}
    </div>
  );
}

function VideoCard({ videoName, persons, onVideoNameClick, onPersonClick, isSelected }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  const sortedPersons = persons.sort((a, b) => a.person_id.localeCompare(b.person_id));

  return (
    <div className={`card ${isSelected ? 'selected' : ''}`}>
      <button className="video-name" onClick={onVideoNameClick}>
        {videoName}
      </button>
      <button className="expand-button" onClick={toggleExpand}>
        {isExpanded ? '접기' : '펼치기'}
      </button>
      {isExpanded && (
        <table className="person-table">
          <tbody>
            {sortedPersons.map((person, index) => (
              <tr key={index} onClick={() => onPersonClick(person)}>
                <td>
                  {person.person_id}<br />
                  나이: {person.person_age}<br />
                  옷: {person.person_clothes}<br />
                  색상: {person.person_color}<br />
                  성별: {person.person_gender}<br />
                  얼굴: <img src={person.person_face} alt={person.person_id} /><br />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default SidebarRight;
