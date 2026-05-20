## 담당 서비스

### 1. 광고 내용 기반, 예산 및 노출 수 예측 (Kiwi, 벡터 유사도)
광고의 제목과 내용을 기반으로 

```java
    @GetMapping("products/{page}")
    public ResponseEntity<?> getRecommends(@PathVariable int page,
                                           @AuthenticationPrincipal CustomUserDetails userDetails) {
        PostProductWithPagingDTO posts = postProductService.getRecommendProducts(page, userDetails.getId());

        posts.getPosts().forEach(post -> {
            if (post.getMemberProfile() != null && !post.getMemberProfile().isBlank()) {
                post.setMemberProfile(
                        toPresignedUrlOrOriginal(post.getMemberProfile())
                );
            }

            if(post.getPostFiles() == null || post.getPostFiles().isEmpty()) {
                return;
            }
            post.setPostFiles(
                    post.getPostFiles().stream()
                            .map(this::toPresignedUrlOrOriginal)
                            .collect(Collectors.toList())
            );

        });

        return ResponseEntity.ok(posts);
    }
```

### 2. 비정형 음성데이터 요약 및 챗봇 (STT-LLM 파이프라인, RAG, LangChain)
광고 페이지는 사이트를 이용하는 모든 회원들이 원하는 광고가 있을 때,
광고를 신청할 수 있는 기능을 제공합니다.


### 3. 게시글 내용 및 첨부파일로 신뢰지수 예측 (NB)



