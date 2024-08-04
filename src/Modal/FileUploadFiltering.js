//FileUploadFiltering.js

import React, { useState, useContext, useEffect } from 'react';
import './FileUploadFiltering.css';
import Swal from 'sweetalert2';
import { DataContext } from '../Data/DataContext'; 

const Server_IP = process.env.REACT_APP_Server_IP;

function FileUploadFiltering() {
  const { data } = useContext(DataContext);
  const [videoFile, setVideoFile] = useState(null);
  const [address, setAddress] = useState('');
  const [camName, setCamName] = useState('');
  const [startTime, setStartTime] = useState('');
  const [list, setList] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [age, setAge] = useState('');
  const [color, setColor] = useState('');
  const [clothes, setClothes] = useState('');
  const [gender, setGender] = useState('');
  const [createClip, setCreateClip] = useState(true);
  const [addressOptions, setAddressOptions] = useState([]); 
  const [filteredCamOptions, setFilteredCamOptions] = useState([]); 

  const userId = data.userId; 

   // 주소 선택 옵션을 설정하기 위한 useEffect
  useEffect(() => {
    if (data.mapCameraInfo && data.mapCameraInfo.length > 0) {
      const uniqueAddresses = [...new Set(data.mapCameraInfo.map(cam => cam.address))];
      setAddressOptions(uniqueAddresses);
    }
  }, [data.mapCameraInfo]);

  // 주소 선택 시 필터된 캠 선택 옵션을 설정하기 위한 useEffect
  useEffect(() => {
    if (address) {
      const cams = data.mapCameraInfo.filter(cam => cam.address === address);
      setFilteredCamOptions(cams);
      setCamName('');   // 주소 변경 시 캠 이름 초기화
    } else {
      setFilteredCamOptions([]);
    }
  }, [address, data.mapCameraInfo]);

  // 리스트에 비디오 파일을 추가하는 이벤트
  const handleAddToList = () => {
    if (!videoFile || !camName || !startTime) {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '모든 필드를 입력하세요.',
      });
      return;
    }

    if (videoFile.type !== 'video/mp4') {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '오직 MP4 파일만 업로드 가능합니다.',
      });
      return;
    }

    const fileName = videoFile.name;
    setList([...list, { file: videoFile, fileName, camName, startTime }]);
    setVideoFile(null);
    setAddress('');
    setCamName('');
    setStartTime('');
    document.getElementById("videoFileInput").value = '';
  };

  // 비디오 파일 선택 시 호출되는 함수
  const handleFileChange = (event) => {
    setVideoFile(event.target.files[0]);
  };

  // 이미지 파일 선택 시 호출되는 함수
  const handleImageFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
    } else {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '이미지 파일만 업로드 가능합니다.',
      });
      event.target.value = ''; // 파일 입력 필드 초기화
    }
  };

   // 비디오파일과 필터링 데이터를 전송하는 이벤트
  const handleSubmit = async () => {
    if (list.length === 0) {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '최소 하나의 비디오 파일을 리스트에 추가하세요.',
      });
      return;
    }

    if (!age && !color && !clothes && !gender) {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '최소 하나의 필터 조건을 설정하세요.',
      });
      return;
    }

    try {
      const videoData = await Promise.all(
        list.map(async (item) => ({
          video_name: item.fileName,
          video_content: await toBase64(item.file),
          cam_name: item.camName,
          start_time: item.startTime,
        }))
      );

      let imageData = {};
      if (selectedFile) {
        imageData = {
          image_name: selectedFile.name,
          image_content: await toBase64(selectedFile),
        };
      }

      const user_data = {
        user_id: userId,
      };

      const filter_data = {
        age: age || null,
        color: color || null,
        type: clothes || null,
        gender: gender || null,
      };

      const flag_data = {
        clip_flag: createClip ? 'true' : 'false',
      };

      const data = {
        video_data: videoData,
        user_data: user_data,
        filter_data: filter_data,
        flag_data: flag_data,
      };

      if (selectedFile) {
        data.image_data = imageData;
      }

      console.log('보낼 JSON:', JSON.stringify(data));

      const response = await fetch(`${Server_IP}/upload_file`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        Swal.fire({
          icon: 'success',
          title: '성공',
          text: '파일이 성공적으로 업로드되었습니다.',
        });
      } else {
        Swal.fire({
          icon: 'error',
          title: '실패',
          text: '파일 업로드에 실패했습니다.',
        });
      }
    } catch (error) {
      console.error('파일 업로드 중 오류 발생:', error);
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '파일 업로드 중 오류가 발생했습니다.',
      });
    }
  };

  // 파일을 Base64로 인코딩하는 함수
  const toBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = (error) => reject(error);
    });
  };

  return (
    <div className="file-upload-filtering">
      <div className="file-upload">
        <h2>파일 업로드</h2>
        <select
          value={address}
          onChange={(e) => setAddress(e.target.value)}
        >
          <option value="">주소 선택</option>
          {addressOptions.map((addr, index) => (
            <option key={index} value={addr}>
              {addr}
            </option>
          ))}
        </select>
        <select
          value={camName}
          onChange={(e) => setCamName(e.target.value)}
          disabled={!address}
        >
          <option value="">캠 이름 선택</option>
          {filteredCamOptions.map((cam, index) => (
            <option key={index} value={cam.cam_name}>
              {cam.cam_name}
            </option>
          ))}
        </select>
        <input
          type="text"
          placeholder="시작 시간"
          value={startTime}
          onChange={(e) => setStartTime(e.target.value)}
        />
        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={createClip}
              onChange={(e) => setCreateClip(e.target.checked)}
            />
            {createClip ? ' 클립 생성' : ' 클립 미생성'}
          </label>
        </div>
        <input
          type="file"
          id="videoFileInput"
          accept="video/mp4"
          onChange={handleFileChange}
        />
        <div className="buttons">
          <button onClick={handleAddToList}>리스트 추가</button>
        </div>
        <ul className="file-list">
          {list.map((item, index) => (
            <li key={index}>
              <span className="file-name">{item.fileName}</span>
              <span className="cam-name">{item.camName}</span>
              <span className="start-time">{item.startTime}</span>
            </li>
          ))}
        </ul>
      </div>
      <div className="filtering">
        <h2>필터링</h2>
        <div className="form-group">
          <label>이미지 파일:</label>
          <input type="file" accept="image/*" onChange={handleImageFileChange} />
        </div>
        <div className="form-group">
          <label>나이:</label>
          <select value={age} onChange={(e) => setAge(e.target.value)}>
            <option value="">선택하지 않음</option>
            <option value="Child">유아</option>
            <option value="Youth">청년</option>
            <option value="Middle">중년</option>
            <option value="Old">노년</option>
          </select>
        </div>
        <div className="form-group">
          <label>옷 색상:</label>
          <select value={color} onChange={(e) => setColor(e.target.value)}>
            <option value="">선택하지 않음</option>
            <option value="black">검정색</option>
            <option value="white">하얀색</option>
            <option value="red">빨간색</option>
            <option value="yellow">노란색</option>
            <option value="green">초록색</option>
            <option value="blue">파란색</option>
            <option value="grey">갈색</option>
          </select>
        </div>
        <div className="form-group">
          <label>상하의 종류:</label>
          <select value={clothes} onChange={(e) => setClothes(e.target.value)}>
            <option value="">선택하지 않음</option>
            <option value="dress">드레스</option>
            <option value="longsleevetop">긴팔상의</option>
            <option value="shortsleevetop">반팔상의</option>
            <option value="vest">조끼</option>
            <option value="shorts">반바지</option>
            <option value="pants">긴바지</option>
            <option value="skirt">치마</option>
          </select>
        </div>
        <div className="form-group">
          <label>성별:</label>
          <div className="radio-group">
            <label>
              <input
                type="radio"
                value="남성"
                checked={gender === '남성'}
                onChange={(e) => setGender(e.target.value)}
              />
              남성
            </label>
            <label>
              <input
                type="radio"
                value="여성"
                checked={gender === '여성'}
                onChange={(e) => setGender(e.target.value)}
              />
              여성
            </label>
          </div>
        </div>
        <button onClick={handleSubmit}>제출하기</button>
      </div>
    </div>
  );
}

export default FileUploadFiltering;

