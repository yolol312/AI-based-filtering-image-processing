import React, { useContext, useEffect, useRef } from "react";
import { LogDataContext } from "../Data/LogDataContext";
import "./Footer.css";

function Footer(sidebarWidth) {
  const { logInfo } = useContext(LogDataContext); // 로그 데이터를 가져옴
  const consoleRef = useRef(null); // 콘솔 요소에 대한 ref 생성

  // 로그가 업데이트될 때마다 스크롤을 맨 아래로 내림
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [logInfo]); // logInfo가 업데이트될 때마다 실행됨

  return (
    <div
      className="Footer"
      style={{ left: `${sidebarWidth}px` }} // Sidebar의 너비만큼 left 설정
    >
      <div className="console" ref={consoleRef}>
        {logInfo.map((log, index) => {
          // log가 null이나 undefined인 경우 아무것도 렌더링하지 않음
          if (!log || !log.timestamp) return null;

          // person_data에서 필요한 정보만 추출
          const personData = log.person_data.split(", ").reduce((acc, item) => {
            const [key, value] = item.split(": ");
            if (
              key === "Track ID" ||
              key === "Gender" ||
              key === "Age" ||
              key === "UpClothes" ||
              key === "DownClothes"
            ) {
              acc[key] = value;
            }
            return acc;
          }, {});

          return (
            <div key={index} className="log-line">
              <p>
                <strong>Timestamp:</strong> {log.timestamp}
              </p>
              <p>
                <strong>Track ID:</strong> {personData["Track ID"]}
                {"    |    "}
                <strong>Gender:</strong> {personData["Gender"]}
                {"    |    "}
                <strong>Age:</strong> {personData["Age"]}
                {"    |    "}
                <strong>UpClothes:</strong> {personData["UpClothes"]}
                {"    |    "}
                <strong>DownClothes:</strong> {personData["DownClothes"]}
              </p>
              {log.image && <img src={log.image} alt="Log Image" />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default Footer;
