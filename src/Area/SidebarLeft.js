// SidebarLeft.js
import React, { useContext, useState, useEffect } from 'react';
import { DataContext } from '../Data/DataContext';
import './SidebarLeft.css';

function SidebarLeft({ onAddressSelect }) { // onAddressSelect prop 추가
  const { data } = useContext(DataContext);
  const [groupedVideos, setGroupedVideos] = useState({});

  useEffect(() => {
    console.log("Loaded data:", data);
    const videos = data.mapCameraInfo || [];
    const grouped = videos.reduce((acc, current) => {
      if (!acc[current.address]) {
        acc[current.address] = [];
      }
      acc[current.address].push(current); // 배열로 저장
      return acc;
    }, {});

    setGroupedVideos(grouped);
  }, [data]);

  const [selectedAddress, setSelectedAddress] = useState(null);

  const handleVideoClick = (address) => {
    setSelectedAddress(address);
    onAddressSelect(groupedVideos[address]); // 선택된 주소의 데이터를 콜백 함수로 전달
  };

  return (
    <div className="SidebarLeft">
      <h2 className="SidebarLefttitle">지도 목록</h2>
      {Object.keys(groupedVideos).map((address, index) => (
        <button
          key={index}
          className={`video-button ${selectedAddress === address ? 'selected' : ''}`}
          onClick={() => handleVideoClick(address)}
        >
          {address}
        </button>
      ))}
    </div>
  );
}

export default SidebarLeft;
