import React, { useState } from "react";
import "./PersonSwap.css";

const PersonSwap = () => {
  const [person, setPerson] = useState("Person1");

  const handlePersonChange = (newPerson) => {
    setPerson(newPerson);
  };

  return (
    <div className="container">
      <div className="sidebar">
        <div className="person-info">
          <h2>{person} 정보</h2>
          <div className="info-box">
            <p>
              <strong>나이:</strong> 청년
            </p>
            <p>
              <strong>옷 종류:</strong> 반팔 상의
            </p>
            <p>
              <strong>성별:</strong> 남자
            </p>
          </div>
          <div className="buttons">
            <button> 보기</button>
            <button>경로 보기</button>
          </div>
        </div>
        <div className="map">
          <img src="map-placeholder.png" alt="Map" />
        </div>
      </div>
      <div className="main-content">
        <video controls>
          <source src="video-placeholder.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
};

export default PersonSwap;
