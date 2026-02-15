# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['human_play.py'],
             pathex=['D:\\AMyPythonProject\\AIGobang\\myGomoku1'],
             binaries=[(r"D:\Anaconda3\envs\python_3.7_pyinstaller\Lib\site-packages\tensorflow_core\lite\experimental\microfrontend\python\ops\_audio_microfrontend_op.so",r'.\tensorflow_core\lite\experimental\microfrontend\python\ops')],
             datas=[],
             hiddenimports=['tensorflow','tensorflow.compat.v1','numpy','tqdm','collections','copy','tkinter'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='human_play',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
